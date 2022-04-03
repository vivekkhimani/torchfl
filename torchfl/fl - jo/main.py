import argparse
from collections import deque
import threading

import Client
import Server
import ServerModel_FL as ServerModel
import ClientModel_FL as ClientModel

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m' # okay server
    OKCYAN = '\033[96m' # okay clients
    OKGREEN = '\033[92m'  # okay main
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def parse_arguments(parser):
    parser.add_argument('-c', '--clients', type=int, 
                        default=2, action='store', 
                        dest='clients', help='count of clients')
    parser.add_argument('-r', '--rounds', type=int,
                        default=100, action='store', 
                        dest='rounds', help='count of rounds')
    parser.add_argument('-C', '-client_model', default=None,
                        action='store', dest='client_model_args',
                        help='Arguments for the client model, given in' +
                        '"arg_name1=value1,arg_name2=value2" format')
    parser.add_argument('-S', '-server_model', default=None,
                        action='store', dest='server_model_args',
                        help='Arguments for the server model, given in' +
                        '"arg_name1=value1,arg_name2=value2" format')
    return parser.parse_args()

def main():

    # parse arguments
    args = parse_arguments(argparse.ArgumentParser())
    clients = args.clients
    rounds = args.rounds
    client_model_args = args.client_model_args
    server_model_args = args.server_model_args


    # Create Clients

    client_list = {} # clientID: (lock_server, message_queue_server, lock_client, message_queue_client)
    client_models = []
    client_threads = []
    # Create clients
    for client in range(clients):
        #Assuming Local infrastractor
        #TODO: option for remote client
        if client_model_args:
            client_model = ClientModel.ClientModel(client, client_model_args)
        else:
            client_model = ClientModel.ClientModel(client)
            
        client_models.append(client_model)

        event_server = threading.Event() # NOTE: client becomes sequential, can receive one task per iteration
        message_q_server = deque()

        lock_client = threading.Lock()
        message_q_client = deque()

        _serve_com = (event_server, message_q_server, lock_client, message_q_client)
        client_thread = Client.Client(client_model, _serve_com, client)
        client_thread.deamon = True #TODO: check if that is needed
        client_thread.start()
        client_threads.append(client_thread)

        client_list.update({client:_serve_com})

    # Create Server

    if server_model_args:
        server_model = ServerModel.ServerModel(server_model_args)
    else:
        server_model = ServerModel.ServerModel()

    end_event = threading.Event()
    server_thread = Server.Server(server_model, client_list, rounds, end_event) #end_event
    server_thread.deamon = True #TODO: check if that is needed
    server_thread.start()

    end_event.wait()
    print(f'{bcolors.OKGREEN} Training ended {bcolors.ENDC}')

    # kill threads
    #TODO

    # check performance
   


if __name__ == '__main__':
    main()