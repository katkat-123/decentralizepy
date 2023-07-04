import importlib
import logging

import torch


class Sharing:
    """
    API defining who to share with and what, and what to do on receiving

    """

    def __init__(
        self,
        rank,
        machine_id,
        communication,
        mapping,
        graph,
        model,
        dataset,
        log_dir,
        compress=False,
        compression_package=None,
        compression_class=None,
        float_precision=None,
    ):
        """
        Constructor

        Parameters
        ----------
        rank : int
            Local rank
        machine_id : int
            Global machine id
        communication : decentralizepy.communication.Communication
            Communication module used to send and receive messages
        mapping : decentralizepy.mappings.Mapping
            Mapping (rank, machine_id) -> uid
        graph : decentralizepy.graphs.Graph
            Graph reprensenting neighbors
        model : decentralizepy.models.Model
            Model to train
        dataset : decentralizepy.datasets.Dataset
            Dataset for sharing data. Not implemented yet! TODO
        log_dir : str
            Location to write shared_params (only writing for 2 procs per machine)

        """
        self.rank = rank
        self.machine_id = machine_id
        self.uid = mapping.get_uid(rank, machine_id)
        self.communication = communication
        self.mapping = mapping
        self.graph = graph
        self.model = model
        self.dataset = dataset
        self.communication_round = 0
        self.log_dir = log_dir

        self.shapes = []
        self.lens = []
        with torch.no_grad():
            for _, v in self.model.state_dict().items():
                self.shapes.append(v.shape)
                t = v.flatten().numpy()
                self.lens.append(t.shape[0])

        self.compress = compress

        if compression_package and compression_class:
            compressor_module = importlib.import_module(compression_package)
            compressor_class = getattr(compressor_module, compression_class)
            self.compressor = compressor_class(float_precision=float_precision)
            logging.debug(f"Using the {compressor_class} to compress the data")
        else:
            assert not self.compress

    def compress_data(self, data):
        result = dict(data)
        if self.compress:
            if "params" in result:
                result["params"] = self.compressor.compress_float(result["params"])
        return result

    def decompress_data(self, data):
        if self.compress:
            if "params" in data:
                data["params"] = self.compressor.decompress_float(data["params"])
        return data

    def serialized_model(self):
        """
        Convert model to a dictionary. Here we can choose how much to share

        Returns
        -------
        dict
            Model converted to dict

        """
        to_cat = []
        with torch.no_grad():
            for _, v in self.model.state_dict().items():    #gets each tensor
                t = v.flatten()
                to_cat.append(t)    
        flat = torch.cat(to_cat)    #concatenates all tensors, so flattens it
        data = dict()
        data["params"] = flat.numpy()   #stores them as a numpy array
        logging.info("Model sending this round: {}".format(data["params"]))
        return self.compress_data(data) #simpiezei ta data, mallon den kanei tipota pros to parwn

    def deserialized_model(self, m):
        """
        Convert received dict to state_dict.

        Parameters
        ----------
        m : dict
            received dict

        Returns
        -------
        state_dict
            state_dict of received

        """
        state_dict = dict()
        m = self.decompress_data(m)
        T = m["params"]
        start_index = 0
        for i, key in enumerate(self.model.state_dict()):
            end_index = start_index + self.lens[i]
            state_dict[key] = torch.from_numpy(
                T[start_index:end_index].reshape(self.shapes[i])
            )
            start_index = end_index

        return state_dict

    def _pre_step(self):
        """
        Called at the beginning of step.

        """
        pass

    def _post_step(self):
        """
        Called at the end of step.

        """
        pass


    def _averaging_ADPSGD(self, data):
        """
        average two models with weight of 0.5 each

        """

        with torch.no_grad():
            total = dict()

            iteration, sender_uid = data["iteration"], data["sender_uid"]
            del data["degree"]
            del data["iteration"]
            del data["sender_uid"]
            del data["CHANNEL"]

            logging.debug("averaging model from neighbor of iteration {} of neighbor {}".format(iteration, sender_uid))

            data = self.deserialized_model(data)
           
            weight = 0.5

            for key, value in data.items():
                total[key] = weight * value

            for key, value in self.model.state_dict().items():
                total[key] += (1 - weight) * value

        self.model.load_state_dict(total)
        self._post_step()
        self.communication_round += 1


    def _averaging_gossip(self, data):
        """
        average w.r.t. model's age

        """

        with torch.no_grad():
            total = dict()

            iteration, sender_age = data["iteration"], data["age"]
            del data["degree"]
            del data["iteration"]
            del data["age"]
            del data["CHANNEL"]

            if sender_age == 0:
                return
            

            logging.debug(
                "Averaging model from neighbor of iteration {} with age {}".format(
                    iteration, sender_age
                )
            )

            data = self.deserialized_model(data) #kanei decompress to modelo tou geitona
            
            weight = sender_age/(sender_age+self.model.age_t)
            
            logging.debug(
                "weight for averaging is {}".format(
                    weight
                )
            )

            for key, value in data.items(): #apothikevei sto total gia kathe value tou geitona epi to varos tou
                total[key] = value * weight

            for key, value in self.model.state_dict().items():  #prosthetei sto total kai to diko tou montelo, me weight oso menei gia na ftasei sto 1
                total[key] += (1 - weight) * value 

            #prepei na orisw to neo age tou topikou komvou ws to max (sender_age, self_age)
            #giati alliws den tha mporesei na ginei swsta h ananewsh varwn, oso pio megalo to age simainei oti toso pio kainourio einai 
            self.model.age_t = max(self.model.age_t, sender_age)

            logging.debug(
                "new age is {}".format(
                    self.model.age_t
                )
            )

        self.model.load_state_dict(total)
        self._post_step()
        self.communication_round += 1


    # def _averaging_gossip(self, msg_deque):
    #     """
    #     follow paper guidelines, average w.r.t. model's age
    #     """

    #     with torch.no_grad():
    #         total = dict()

    #         data=msg_deque.popleft()
    #         iteration, sender_age = data["iteration"], data["age"]
    #         del data["degree"]
    #         del data["iteration"]
    #         del data["age"]
    #         del data["CHANNEL"]

    #         logging.debug(
    #             "Averaging model from neighbor of iteration {}".format(
    #                 iteration
    #             )
    #         )

    #         data = self.deserialized_model(data) #kanei decompress to modelo tou geitona

    #         weight = sender_age/(sender_age+self.model.age_t)

    #         for key, value in data.items(): #apothikevei sto total gia kathe value tou geitona epi to varos tou
    #             total[key] = value * weight

    #         for key, value in self.model.state_dict().items():  #prosthetei sto total kai to diko tou montelo, me weight oso menei gia na ftasei sto 1
    #             total[key] += (1 - weight) * value 

    #         #prepei na orisw to neo age tou topikou komvou ws to max (sender_age, self_age)
    #         #giati alliws den tha mporesei na ginei swsta h ananewsh varwn, oso pio megalo to age simainei oti toso pio kainourio einai 
    #         self.model.age_t = max(self.model.age_t, sender_age)

    #     self.model.load_state_dict(total)
    #     self._post_step()
    #     self.communication_round += 1


    def _averaging_gossip_queue(self, gossip_queue):
        """
        Averages the local model with all the received models

        """

        if gossip_queue.empty():
            return 0


        with torch.no_grad():
            total = dict()

            queue_size = gossip_queue.qsize()
            
            max_age = self.model.age_t
            ages_sum = self.model.age_t
            # for i in range(queue_size):
            #     sender_age = gossip_queue.get(i)["age"]
                
        
            for i in range(queue_size): #gia kathe geitona
                data = gossip_queue.get(i)
                iteration, sender_age, rank, machine_id = data["iteration"], data["age"], data["rank"], data["machine_id"]
                del data["degree"]
                del data["rank"]
                del data["machine_id"]
                del data["age"]
                del data["iteration"]
                del data["CHANNEL"]
                logging.debug(
                    "Averaging model from neighbor {} of machine {} of iteration {}".format(
                        rank, machine_id, iteration
                    )
                )

                max_age = sender_age if sender_age > max_age else max_age

                data = self.deserialized_model(data) #kanei decompress to modelo tou geitona
                
                ages_sum += sender_age

                for key, value in data.items(): #apothikevei sto total gia kathe value tou geitona epi to varos tou
                    if key in total:
                        total[key] += value * sender_age
                    else:
                        total[key] = value * sender_age

            for key, value in self.model.state_dict().items():  #prosthetei sto total kai to diko tou montelo
                if key in total:
                    total[key] += (self.model.age_t) * value  
                else:
                    total[key] = (self.model.age_t) * value


            for key in total: 
                total[key] /= ages_sum

            self.model.age_t = max_age

        self.model.load_state_dict(total)
        self._post_step()
        self.communication_round += 1 #xreiazetai afto?

        return queue_size


    def _averaging(self, peer_deques):
        """
        Averages the received model with the local model

        """
        with torch.no_grad():
            total = dict()
            weight_total = 0
            for i, n in enumerate(peer_deques): #gia kathe geitona
                data = peer_deques[n].popleft()
                degree, iteration = data["degree"], data["iteration"]
                del data["degree"]
                del data["iteration"]
                del data["CHANNEL"]
                logging.debug(
                    "Averaging model from neighbor {} of iteration {}".format(
                        n, iteration
                    )
                )
                data = self.deserialized_model(data) #kanei decompress to modelo tou geitona
                # Metro-Hastings
                weight = 1 / (max(len(peer_deques), degree) + 1)
                weight_total += weight
                for key, value in data.items(): #apothikevei sto total gia kathe value tou geitona epi to varos tou
                    if key in total:
                        total[key] += value * weight
                    else:
                        total[key] = value * weight

            for key, value in self.model.state_dict().items():  #prosthetei sto total kai to diko tou montelo, me weight oso menei gia na ftasei sto 1
                total[key] += (1 - weight_total) * value  # Metro-Hastings

        self.model.load_state_dict(total)
        self._post_step()
        self.communication_round += 1


    def get_data_to_send(self, degree=None):
        self._pre_step()
        data = self.serialized_model()  #ftiaxnei tous tensores se ena dictionary -> data["parameters"]={compressed data}
        my_uid = self.mapping.get_uid(self.rank, self.machine_id)
        data["degree"] = degree if degree != None else len(self.graph.neighbors(my_uid))
        data["iteration"] = self.communication_round
        return data

    def _averaging_server(self, peer_deques):
        """
        Averages the received models of all working nodes

        """
        with torch.no_grad():
            total = dict()
            for i, n in enumerate(peer_deques):
                data = peer_deques[n].popleft()
                degree, iteration = data["rank"], data["iteration"]
                del data["degree"]
                del data["iteration"]
                del data["CHANNEL"]
                logging.debug(
                    "Averaging model from neighbor {} of iteration {}".format(
                        n, iteration
                    )
                )
                data = self.deserialized_model(data)
                weight = 1 / len(peer_deques)
                for key, value in data.items():
                    if key in total:
                        total[key] += weight * value
                    else:
                        total[key] = weight * value

        self.model.load_state_dict(total)
        self._post_step()
        self.communication_round += 1
        return total
