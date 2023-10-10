==============================
Asynchronous algorithms for DL
==============================

The main script is run.sh, that enables you to run the following asynchronous algorithms for decentralized learning:

* ADPSGD
* Gossip Learning
* Our version of Gossip Learning
  
To execute the corresponding algorithm, modify the ``testing_{adpsgd, gossip, our_gossip}.py`` in the run.sh file.

-------------------------
Before running the code
-------------------------

In the config.ini in [SHARING] section, set the following parameters:

* For ADPSGD: ::
  
    sharing_package = decentralizepy.sharing.AsyncSharing.ADPSGDSharing
    sharing_class = ADPSGDSharing

* For Gossip Learning and Our Gossip Learning: ::
  
    sharing_package = decentralizepy.sharing.AsyncSharing.GossipSharing
    sharing_class = GossipSharing


Important parameters to set in the run.sh file:

* ``training_time``: the amount of time in minutes that the algorithm should run for
* ``timeout``: time in seconds, used only in original gossip learning
* ``test_after``: number of rounds after which the model is saved or evaluated on the test set
* ``save_no_test``: if set to True (default), the model is saved but not evaluated on the test set


**Important**: For ADPSGD experiments, the graph should be bipartite.

-------------------------
Evaluation on test set
-------------------------

In the read_and_test folder there is a python script for reading the saved models and evaluating them on the test set of CIFAR10 dataset.
After the execution of read_and_test.py, a json file is generated for each node containing test accuracy and test loss with respect to iterations and time.


