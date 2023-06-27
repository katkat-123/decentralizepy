import importlib
import json
import logging
import math
import os
import time
import random
import threading
from threading import Thread, Lock
from collections import deque

import torch
from matplotlib import pyplot as plt

from decentralizepy import utils
from decentralizepy.graphs.Graph import Graph
from decentralizepy.mappings.Mapping import Mapping
from decentralizepy.node.Node import Node


class GossipNode(Node):
    """
    This class defines the node for gossip learning algorithm

    """


    def save_plot(self, l, label, title, xlabel, filename):
        """
        Save Matplotlib plot. Clears previous plots.

        Parameters
        ----------
        l : dict
            dict of x -> y. `x` must be castable to int.
        label : str
            label of the plot. Used for legend.
        title : str
            Header
        xlabel : str
            x-axis label
        filename : str
            Name of file to save the plot as.

        """
        plt.clf()
        y_axis = [l[key] for key in l.keys()]
        x_axis = list(map(int, l.keys()))
        plt.plot(x_axis, y_axis, label=label)
        plt.xlabel(xlabel)
        plt.title(title)
        plt.savefig(filename)

    def get_neighbors(self, node=None):
        return self.my_neighbors

    def receive_GOSSIP(self):
        return self.receive_channel("GOSSIP", False)

    
    def send_gossip_to_peers(self):
        
        while(True):
            
            logging.debug("KAT-thread : in send gossip before waiting")

            time.sleep(self.delta_g) #in seconds


            #to vazw mesa sth loopa gia na elegxei kathe fora an einai zwntanoi oi geitones tou - den kserw an vgazei poli noima aftos o kwdikas
            new_neighbors = self.get_neighbors() 
            self.my_neighbors = new_neighbors
            self.connect_neighbors()
            logging.debug("KAT-thread : connected to all neighbors")

            #send to peers
            neighbor_list = list(self.my_neighbors)
            peer_to_send = random.choice(neighbor_list)

            self.model_lock.acquire()
            to_send = self.sharing.get_data_to_send(degree=len(self.my_neighbors))  #epistrefei ena dictionary me pedia: parameters, degree, iteration + prepei na valw kai to age
            self.model_lock.release()

            to_send["CHANNEL"] = "GOSSIP"
            to_send["age"] = self.model.age_t


            self.communication.send(peer_to_send, to_send)
        
            logging.info("KAT-thread: Sent to neighbor in thread with age {}".format(to_send["age"]))

            if self.stopper.is_set():
                logging.info("KAT-thread: Teleiwsa")
                break

            
        #after signal is set, before terminating the thread should i do sth (save, safe exit etc.)
        logging.info("KAT-thread: Out of loop")


    def run(self):
        """
        Start the decentralized learning

        """


        self.testset = self.dataset.get_testset()
        rounds_to_test = self.test_after
        rounds_to_train_evaluate = self.train_evaluate_after
        global_epoch = 1
        change = 1


        #create thread for sending periodically the local model to neighbors
        self.stopper = threading.Event()
        sending_thread = threading.Thread(target=self.send_gossip_to_peers)
        logging.debug("KAT: Created thread")
        sending_thread.start()
        
        self.model_lock = Lock()

    
        t_end = time.time() + 60 * self.training_time

        iteration = 0

        # tr_time = 0
        # max_time = 0

        while(True):

            if time.time() >= t_end:    #stop the training after a specific amount of time
                break

            try:
                sender, data = self.receive_GOSSIP() #receive from neighbor
            except TypeError:   #epestrepse none=xtipise timeout, ara ksanampes sti loopa
                continue

            logging.debug(
                    "Received Model from {} of iteration {}".format(    #to iteration exei noima? afou einai gossip
                        sender, data["iteration"]
                    )
            )            

            # self.msg_deque.append(data)

            rounds_to_train_evaluate -= 1
            rounds_to_test -= 1
            self.iteration = iteration

            
            self.model_lock.acquire()
            # start = time.time()
            self.sharing._averaging_gossip(data)  
            self.trainer.train(self.dataset)
            # end = time.time()
            # tr_time += end-start
            # max_time = (end-start) if (end-start)>max_time else max_time
            self.model_lock.release()



            if self.reset_optimizer:    #kanei reset ton optimizer, an mas to leei to config
                self.optimizer = self.optimizer_class(
                    self.model.parameters(), **self.optimizer_params
                )  # Reset optimizer state
                self.trainer.reset_optimizer(self.optimizer)

            if iteration:   #saving the metrics
                with open(
                    os.path.join(self.log_dir, "{}_results.json".format(self.rank)),
                    "r",
                ) as inf:
                    results_dict = json.load(inf)
            else:
                results_dict = {
                    "train_loss": {},
                    "test_loss": {},
                    "test_acc": {},
                    "total_bytes": {},
                    "total_meta": {},
                    "total_data_per_n": {},
                }

            results_dict["total_bytes"][iteration + 1] = self.communication.total_bytes

            if hasattr(self.communication, "total_meta"):   #saving more metadata?
                results_dict["total_meta"][
                    iteration + 1
                ] = self.communication.total_meta
            if hasattr(self.communication, "total_data"):
                results_dict["total_data_per_n"][
                    iteration + 1
                ] = self.communication.total_data


            if rounds_to_train_evaluate == 0:   #evaluating after fixed size of iterations
                logging.info("Evaluating on train set.")
                rounds_to_train_evaluate = self.train_evaluate_after * change
                loss_after_sharing = self.trainer.eval_loss(self.dataset)   #trainer eval loss -> check
                results_dict["train_loss"][iteration + 1] = loss_after_sharing
                self.save_plot(
                    results_dict["train_loss"],
                    "train_loss",
                    "Training Loss",
                    "Communication Rounds",
                    os.path.join(self.log_dir, "{}_train_loss.png".format(self.rank)),
                )

            if self.dataset.__testing__ and rounds_to_test == 0:    #evaluating on test set
                rounds_to_test = self.test_after * change
                logging.info("Evaluating on test set.")
                ta, tl = self.dataset.test(self.model, self.loss)   #to test ginetai sto module dataset
                results_dict["test_acc"][iteration + 1] = ta
                results_dict["test_loss"][iteration + 1] = tl

                if global_epoch == 49:
                    change *= 2

                global_epoch += change

            with open(
                os.path.join(self.log_dir, "{}_results.json".format(self.rank)), "w"
            ) as of:
                json.dump(results_dict, of)


            iteration += 1

            
        #terminate the sending thread
        logging.info("KAT: time passed - waiting for thread to join")
        self.stopper.set()
        sending_thread.join() 

        # print("node ", self.uid, " with avg tr time ", tr_time/iteration)
        # print("node ", self.uid, " with max tr time ", max_time)

    
        if self.model.shared_parameters_counter is not None:
            logging.info("Saving the shared parameter counts")
            with open(
                os.path.join(
                    self.log_dir, "{}_shared_parameters.json".format(self.rank)
                ),
                "w",
            ) as of:
                json.dump(self.model.shared_parameters_counter.numpy().tolist(), of)
        
        self.disconnect_neighbors() #meta ta iterations, aposindesi apo geitones
        logging.info("Storing final weight")
        self.model.dump_weights(self.weights_store_dir, self.uid, self.iteration)    #apothikefsh twn weights sto modelo
        logging.info("All neighbors disconnected. Process complete!")

    def cache_fields(
        self,
        rank,
        machine_id,
        mapping,
        graph,
        iterations,
        log_dir,
        weights_store_dir,
        test_after,
        train_evaluate_after,
        reset_optimizer,
    ):
        """
        Instantiate object field with arguments.

        Parameters
        ----------
        rank : int
            Rank of process local to the machine
        machine_id : int
            Machine ID on which the process in running
        mapping : decentralizepy.mappings
            The object containing the mapping rank <--> uid
        graph : decentralizepy.graphs
            The object containing the global graph
        iterations : int
            Number of iterations (communication steps) for which the model should be trained
        log_dir : str
            Logging directory
        weights_store_dir : str
            Directory in which to store model weights
        test_after : int
            Number of iterations after which the test loss and accuracy arecalculated
        train_evaluate_after : int
            Number of iterations after which the train loss is calculated
        reset_optimizer : int
            1 if optimizer should be reset every communication round, else 0
        """
        self.rank = rank
        self.machine_id = machine_id
        self.graph = graph
        self.mapping = mapping
        self.uid = self.mapping.get_uid(rank, machine_id)
        self.log_dir = log_dir
        self.weights_store_dir = weights_store_dir
        self.iterations = iterations
        self.test_after = test_after
        self.train_evaluate_after = train_evaluate_after
        self.reset_optimizer = reset_optimizer
        self.sent_disconnections = False

        logging.debug("Rank: %d", self.rank)
        logging.debug("type(graph): %s", str(type(self.rank)))
        logging.debug("type(mapping): %s", str(type(self.mapping)))

    def init_comm(self, comm_configs):
        """
        Instantiate communication module from config.

        Parameters
        ----------
        comm_configs : dict
            Python dict containing communication config params

        """
        comm_module = importlib.import_module(comm_configs["comm_package"])
        comm_class = getattr(comm_module, comm_configs["comm_class"])
        comm_params = utils.remove_keys(comm_configs, ["comm_package", "comm_class"])
        self.addresses_filepath = comm_params.get("addresses_filepath", None)
        self.communication = comm_class(
            self.rank, self.machine_id, self.mapping, self.graph.n_procs, **comm_params
        )

    def instantiate(
        self,
        rank: int,
        machine_id: int,
        mapping: Mapping,
        graph: Graph,
        config,
        iterations=1,
        log_dir=".",
        weights_store_dir=".",
        log_level=logging.INFO,
        test_after=5,
        train_evaluate_after=1,
        reset_optimizer=1,
        delta_g=1, #kat: new add
        training_time=5, #in minutes, kat: new add
        *args
    ):
        """
        Construct objects.

        Parameters
        ----------
        rank : int
            Rank of process local to the machine
        machine_id : int
            Machine ID on which the process in running
        mapping : decentralizepy.mappings
            The object containing the mapping rank <--> uid
        graph : decentralizepy.graphs
            The object containing the global graph
        config : dict
            A dictionary of configurations.
        iterations : int
            Number of iterations (communication steps) for which the model should be trained
        log_dir : str
            Logging directory
        weights_store_dir : str
            Directory in which to store model weights
        log_level : logging.Level
            One of DEBUG, INFO, WARNING, ERROR, CRITICAL
        test_after : int
            Number of iterations after which the test loss and accuracy arecalculated
        train_evaluate_after : int
            Number of iterations after which the train loss is calculated
        reset_optimizer : int
            1 if optimizer should be reset every communication round, else 0
        args : optional
            Other arguments

        """
        logging.info("Started process.")

        self.init_log(log_dir, rank, log_level)

        self.cache_fields(
            rank,
            machine_id,
            mapping,
            graph,
            iterations,
            log_dir,
            weights_store_dir,
            test_after,
            train_evaluate_after,
            reset_optimizer,
        )
        self.init_dataset_model(config["DATASET"])
        self.init_optimizer(config["OPTIMIZER_PARAMS"])
        self.init_trainer(config["TRAIN_PARAMS"])
        self.init_comm(config["COMMUNICATION"])

        self.message_queue = dict()

        self.barrier = set()
        self.my_neighbors = self.graph.neighbors(self.uid)

        self.init_sharing(config["SHARING"])
        self.msg_deque = deque()
        self.connect_neighbors()

        self.delta_g = delta_g  #kat
        self.training_time = training_time #kat
        self.model.age_t = 0    #kat: tairiazei pio polu na einai apothikevmeno sto model


    def __init__(   #isws prepei na pernei kai alles parametrous
        self,
        rank: int,
        machine_id: int,
        mapping: Mapping,
        graph: Graph,
        config,
        iterations=1,
        log_dir=".",
        weights_store_dir=".",
        log_level=logging.INFO,
        test_after=5,
        train_evaluate_after=1,
        reset_optimizer=1,
        delta_g=0.1, #kat: new add
        training_time=5, #in minutes, kat: new add
        *args
    ):
        """
        Constructor

        Parameters
        ----------
        rank : int
            Rank of process local to the machine
        machine_id : int
            Machine ID on which the process in running
        mapping : decentralizepy.mappings
            The object containing the mapping rank <--> uid
        graph : decentralizepy.graphs
            The object containing the global graph
        config : dict
            A dictionary of configurations. Must contain the following:
            [DATASET]
                dataset_package
                dataset_class
                model_class
            [OPTIMIZER_PARAMS]
                optimizer_package
                optimizer_class
            [TRAIN_PARAMS]
                training_package = decentralizepy.training.Training
                training_class = Training
                epochs_per_round = 25
                batch_size = 64
        iterations : int
            Number of iterations (communication steps) for which the model should be trained
        log_dir : str
            Logging directory
        weights_store_dir : str
            Directory in which to store model weights
        log_level : logging.Level
            One of DEBUG, INFO, WARNING, ERROR, CRITICAL
        test_after : int
            Number of iterations after which the test loss and accuracy arecalculated
        train_evaluate_after : int
            Number of iterations after which the train loss is calculated
        reset_optimizer : int
            1 if optimizer should be reset every communication round, else 0
        args : optional
            Other arguments

        """

        total_threads = os.cpu_count()
        self.threads_per_proc = max(
            math.floor(total_threads / mapping.get_local_procs_count()), 1
        )
        torch.set_num_threads(self.threads_per_proc)
        torch.set_num_interop_threads(1)
        self.instantiate(
            rank,
            machine_id,
            mapping,
            graph,
            config,
            iterations,
            log_dir,
            weights_store_dir,
            log_level,
            test_after,
            train_evaluate_after,
            reset_optimizer,
            delta_g,
            training_time,
            *args
        )
        logging.info(
            "Each proc uses %d threads out of %d.", self.threads_per_proc, total_threads
        )
        self.run()
