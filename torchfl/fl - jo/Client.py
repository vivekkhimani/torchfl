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

class Client(Thread):
    def __init__(self, clientModel, server_, myID):
        Thread.__init__(self)
        self.clientModel = clientModel
        
        self.event_server = server_[0]
        self.message_q_server = server_[1]
        self.lock_client = server_[2]
        self.message_q_client = server_[3]
        
        self.myID = myID

    def run(self):
        print(f'{bcolors.OKCYAN} Client:{self.myID} started successfully!{bcolors.ENDC}')

        self.clientModel.create_model()
        self.clientModel.load_data()

        while(True):
            self.event_server.wait() # wait for task
            task = self.message_q_server.popleft()
            
            if task == 'END':
                return

            res = self.clientModel.new_task(task)
            
            # end task
            self.lock_client.acquire()
            self.message_q_client.append(res)
            self.lock_client.release()

            self.event_server.clear()