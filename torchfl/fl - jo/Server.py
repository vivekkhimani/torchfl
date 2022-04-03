from threading import Thread

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


class Server(Thread):
    def __init__(self, serverModel_, client_list, rounds, end_event):
        Thread.__init__(self)
        self.serverModel = serverModel_
        self.client_list = client_list
        self.rounds = rounds
        self.end_event = end_event
    
    def run(self):
        print(f'{bcolors.OKBLUE} Server started successfully!{bcolors.ENDC}')

        self.serverModel.create_model(self.rounds)
        self.serverModel.reg_clients(self.client_list) # NOTE: Assuminng they have permanent connection TODO: This should change

        print(f'{bcolors.OKBLUE} Start training...{bcolors.ENDC}')

        for epoch in range(self.rounds):
            print(f'{bcolors.OKBLUE}----- Epoch {epoch} -----{bcolors.ENDC}')
            self.serverModel.runEpoch()
        
        print(f'{bcolors.OKBLUE}Start testing{bcolors.ENDC}')
        self.serverModel.test()