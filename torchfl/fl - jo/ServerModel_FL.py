'''
Federated Learning for FEMNIST
'''
import numpy as np
import ClientModel_FL

def update(updates):
    total_weight = 0.
    base = {}
    keys = updates[0][1].keys()
    
    for key in keys:   
        base.update({key:np.zeros(updates[0][1][key].shape)})
    
    for (client_samples, client_model) in updates:
        total_weight += client_samples
        for key in keys:
            base[key] = np.add(base[key],(client_samples * client_model[key]))


    averaged_soln = {}
    for key in keys:
        avg = base[key]/total_weight
        averaged_soln.update({key:avg})

    model = averaged_soln
    return model


class ServerModel:
    def __init__(self):
        pass
    
    def create_model(self, rounds):
        self.client_model = ClientModel_FL.myModel(28,62)
        self.model_params = self.client_model.state_dict()

        
    def reg_clients(self, clients_list): #can be called periodically
        self.clients = clients_list

    def runEpoch(self):
        updates = []

        for client in self.clients:
            msg = {'0':'new', '1':self.model_params}
            self.clients[client][1].append(msg)
            self.clients[client][0].set()

        #wait updates
        return_clients = len(self.clients)
        while(return_clients != 0):
            for i in range(len(self.clients)):
                self.clients[i][2].acquire()
                if self.clients[i][3]:
                    msg = self.clients[i][3].popleft()
                    updates.append((msg['samples'],msg['weights']))
                    return_clients = return_clients - 1
                self.clients[i][2].release()

        
        self.model_params = update(updates)
    
    def test(self):
        for client in self.clients:
            msg = {'0':'test', '1':self.model_params}
            self.clients[client][1].append(msg)
            self.clients[client][0].set()