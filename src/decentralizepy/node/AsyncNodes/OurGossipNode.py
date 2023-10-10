import importlib
import json
import logging
import math
import os
import time
import random
import threading
import csv
from queue import Queue
from datetime import datetime


import torch
from matplotlib import pyplot as plt

from decentralizepy import utils
from decentralizepy.graphs.Graph import Graph
from decentralizepy.mappings.Mapping import Mapping
from decentralizepy.node.Node import Node


class OurGossipNode(Node):
    """
    This class defines the node for our version of the gossip learning algorithm

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

    def receive_and_append(self):
        """
        Function that runs in parallel with training (main thread).
        Receives models from neighbors and appends them to a queue.

        """

        while(True):

            if self.stopper.is_set():    #main thread finished training
                logging.debug("receiver-thread: out of loop")
                break

            try:
                sender, data = self.receive_GOSSIP() 
            except TypeError:   #in case of timeout
                continue

            logging.debug(
                    "Received Model from {} of iteration {}".format(
                        sender, data["iteration"]
                    )
            )            

            self.msg_queue.put(data)

            
        logging.debug("receiver-thread: finished")


    def run(self):
        """
        Runs the training loop.

        """

        self.testset = self.dataset.get_testset()
        rounds_to_test = self.test_after
        rounds_to_train_evaluate = self.train_evaluate_after


        self.stopper = threading.Event()
        
        #create thread for receiving neighbors' models
        receiver_thread = threading.Thread(target=self.receive_and_append)
        logging.debug("Receiver thread created")
        receiver_thread.start()
        
        t_end = time.time() + 60 * self.training_time

        iteration = 0


        while(True):
            
            if time.time() >= t_end:    #stop the training after training_time minutes
                break
            
            logging.info("Starting training iteration: %d", iteration)
            rounds_to_train_evaluate -= 1
            rounds_to_test -= 1
            self.iteration = iteration


            #train
            self.trainer.train(self.dataset)
            
            #update age
            if(not self.trainer.full_epochs):
                self.age_t += self.trainer.rounds * self.trainer.batch_size
            else:
                #correct only for CIFAR10
                self.age_t += self.trainer.rounds * (self.dataset.sizes[self.dataset.dataset_id]*50000)

            #aggregate
            new_age, no_of_aggr_msgs = self.sharing._averaging_gossip_queue(self.msg_queue, self.age_t)
            self.age_t = new_age
            self.msg_aggr += no_of_aggr_msgs


            new_neighbors = self.get_neighbors()
            self.my_neighbors = new_neighbors
            self.connect_neighbors()
            logging.debug("Connected to all neighbors")

            neighbors_list = list(self.my_neighbors)
            peer_to_send = random.choice(neighbors_list)

            to_send = self.sharing.get_data_to_send(self.iteration, self.age_t)
            to_send["CHANNEL"] = "GOSSIP"

            #send
            self.communication.send(peer_to_send, to_send)
            self.msg_sent += 1
        
            
            #store meta info
            self.sending_log.append({"timestamp": datetime.fromtimestamp(datetime.now().timestamp()), "iteration": iteration+1, "from": self.uid, "to": peer_to_send, "age": self.age_t})
        
            if self.reset_optimizer: 
                self.optimizer = self.optimizer_class(
                    self.model.parameters(), **self.optimizer_params
                ) 
                self.trainer.reset_optimizer(self.optimizer)

            if iteration:
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
                    "aggregated_msgs": {}
                }

            results_dict["total_bytes"][iteration + 1] = self.communication.total_bytes            
            results_dict["aggregated_msgs"][iteration + 1] = no_of_aggr_msgs


            if hasattr(self.communication, "total_meta"):
                results_dict["total_meta"][
                    iteration + 1
                ] = self.communication.total_meta
            if hasattr(self.communication, "total_data"):
                results_dict["total_data_per_n"][
                    iteration + 1
                ] = self.communication.total_data


            if rounds_to_train_evaluate == 0 and self.eval_on_train:

                logging.info("Evaluating on train set.")
                rounds_to_train_evaluate = self.train_evaluate_after
                loss_after_sharing = self.trainer.eval_loss(self.dataset)
                results_dict["train_loss"][iteration + 1] = loss_after_sharing

                self.save_plot(
                    results_dict["train_loss"],
                    "train_loss",
                    "Training Loss",
                    "Communication Rounds",
                    os.path.join(self.log_dir, "{}_train_loss.png".format(self.rank)),
                )


            if self.dataset.__testing__ and rounds_to_test == 0:
                rounds_to_test = self.test_after
                self.msg_stats["msg_sent"][iteration + 1] = self.msg_sent
                self.msg_stats["msg_aggr"][iteration + 1] = self.msg_aggr
                
                if self.save_not_test:
                    
                    logging.info("Saving model.")
                    if not os.path.exists(os.path.join(self.log_dir, "models")):
                        os.makedirs(os.path.join(self.log_dir, "models"))

                    torch.save(self.model.state_dict(), os.path.join(self.log_dir, "models/{}_model_{}_iter.pt".format(self.uid, iteration+1)))
                   
                else:
                    
                    logging.info("Evaluating on test set.")
                    ta, tl = self.dataset.test(self.model, self.loss)  
                    results_dict["test_acc"][iteration + 1] = ta
                    results_dict["test_loss"][iteration + 1] = tl
                

            with open(
                os.path.join(self.log_dir, "{}_results.json".format(self.rank)), "w"
            ) as of:
                json.dump(results_dict, of)

            iteration += 1

            
        logging.info("Time passed - waiting for receiver thread to join")
        self.stopper.set()
        receiver_thread.join() 
    
        if self.model.shared_parameters_counter is not None:
            logging.info("Saving the shared parameter counts")
            with open(
                os.path.join(
                    self.log_dir, "{}_shared_parameters.json".format(self.rank)
                ),
                "w",
            ) as of:
                json.dump(self.model.shared_parameters_counter.numpy().tolist(), of)
        

        sending_log_header = ["timestamp", "iteration", "from", "to", "age"]
        with open(os.path.join(self.log_dir, "{}_sending_log.csv".format(self.rank)), 'w+') as file:
            writer = csv.DictWriter(file, fieldnames=sending_log_header)
            writer.writeheader()
            writer.writerows(self.sending_log)


        with open(
                os.path.join(self.log_dir, "{}_msg_stats.json".format(self.uid)), "w+"
            ) as of:
                json.dump(self.msg_stats, of)


        logging.info("Training complete. Disconnecting neighbors.")
        self.disconnect_neighbors()
        logging.info("Storing final weight")
        self.model.dump_weights(self.weights_store_dir, self.uid, iteration)
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
        training_time=5, #in minutes
        save_not_test=True,
        eval_on_train=False,
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
            Number of iterations after which the test loss and accuracy are calculated
        train_evaluate_after : int
            Number of iterations after which the train loss is calculated
        reset_optimizer : int
            1 if optimizer should be reset every communication round, else 0
        training_time : int
            Duration of training in minutes 
        save_not_test : bool
            True if the model should be saved each test_after rounds
            False if the model should be evaluated on the test set each test_after rounds
        eval_on_train : bool
            True if the model should be evaluated on the train set
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
        self.msg_queue = Queue()
        self.connect_neighbors()

        self.sending_log = []
        self.training_time = training_time 
        self.age_t = 0
        self.msg_sent = 0
        self.msg_aggr = 0

        self.msg_stats = {"msg_sent": {}, "msg_aggr": {}}

        self.save_not_test = save_not_test
        self.eval_on_train = eval_on_train

    def __init__(
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
        training_time=5, #in minutes
        save_not_test=True,
        eval_on_train=False,
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
        training_time : int
            Duration of training in minutes 
        save_not_test : bool
            True if the model should be saved each test_after rounds
            False if the model should be evaluated on the test set each test_after rounds
        eval_on_train : bool
            True if the model should be evaluated on the train set
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
            training_time,
            save_not_test,
            eval_on_train,
            *args
        )
        logging.info(
            "Each proc uses %d threads out of %d.", self.threads_per_proc, total_threads
        )
        self.run()
