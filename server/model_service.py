import torch
import torch_geometric as pyg
import torch_scatter
import numpy as np
from model.learning_simulator import LearningSimulator


class ModelService():
    def __init__(self, path_to_model: str):
        self.model : LearningSimulator = LearningSimulator(128, 10, 16, 3, 5)
        model_weights = torch.load(path_to_model)
        self.model.load_state_dict(model_weights['model'])
        self.model.eval()

    
    def predict(self, particles_data : torch.Tensor, position_data : torch.Tensor):
        acc_mean, acc_std = self.calculate_main_statistics(position_data.numpy())
        with torch.no_grad():
            graph = self.preprocess(particles_data, position_data[:, :-1], None, acc_mean, acc_std, 0.0)
            output = self.model(graph)
        return self.apply_acceleration(output, position_data, acc_mean, acc_std)
    
    def apply_acceleration(self, accelerations : torch.Tensor, positions: torch.Tensor, acc_mean : torch.Tensor, acc_std: torch.Tensor):
        current_pos = positions[:, -1]
        prev_pos = positions[:, -2]
        accelerations = accelerations * acc_std + acc_mean
        current_velocity = current_pos - prev_pos
        pred_velocity = current_velocity + accelerations
        return current_pos + pred_velocity


    def calculate_main_statistics(self, positions: np.ndarray) -> torch.Tensor:
        velocities = positions[:, 1:] - positions[:, :-1]
        accelerations = velocities[:, 1:] - velocities[:, :-1]
        acc_mean = torch.tensor(np.mean(accelerations, axis=(0,1)))
        acc_std = torch.tensor(np.std(accelerations, axis = (0,1)))
        return acc_mean, acc_std

    def preprocess(self, node_features, pos_seq, target_pos, acc_mean, acc_std, noise_std):
        #apply noise
        pos_noise = self.generate_noise(pos_seq, noise_std)
        pos_seq = pos_seq + pos_noise

        #calculcate velocities
        current_pos = pos_seq[:, -1, :]
        vel_seq = pos_seq[:, 1:, :] - pos_seq[:, :-1, :]

        #construct the graph based on the distances between particles
        n_particle = current_pos.size(0)
        edge_index = pyg.nn.radius_graph(current_pos, 1000, loop=True, max_num_neighbors = n_particle)

        dim = current_pos.size(-1)
        edge_displacement = (torch.gather(current_pos, dim = 0, index=edge_index[0].unsqueeze(-1).expand(-1, dim)) - torch.gather(current_pos, dim = 0, index=edge_index[1].unsqueeze(-1).expand(-1, dim)))
        edge_displacement /= 1000
        edge_distance = torch.norm(edge_displacement, dim=-1, keepdim=True)

        #add bounderies handling in case of water
        
        if target_pos is not None:
            prev_velocity = vel_seq[:, -1, :]
            next_velocity = target_pos + pos_noise[:, -1, :] - current_pos
            acceleration = next_velocity - prev_velocity
            acceleration = (acceleration - acc_mean) / torch.sqrt(acc_std ** 2 + noise_std ** 2) 
        else:
            acceleration = None
        
        graph = pyg.data.Data(
            x = node_features,
            edge_index = edge_index,
            edge_attr = torch.cat((edge_displacement, edge_distance), dim = -1),
            y = acceleration,
            pos = vel_seq.reshape(vel_seq.size(0), -1)
        )
        return graph
    
    def generate_noise(self, position_seq, noise_std):
        velocity_seq = position_seq[:, 1:, :] - position_seq[:, :-1, :]
        time_steps = velocity_seq.size(1)
        velocity_noise = torch.randn_like(velocity_seq) * (noise_std / time_steps ** 0.5)
        velocity_noise = velocity_noise.cumsum(dim=1)
        pos_noise = velocity_noise.cumsum(dim=1)
        pos_noise = torch.cat((torch.zeros_like(pos_noise)[:,0:1], pos_noise), dim = 1)
        return pos_noise
    
    def preprocess_raw_data(self, particle_data : list, positions : list):
        particle_data = np.array(particle_data)
        particle_data = torch.from_numpy(particle_data).to(torch.int)
        
        positions = np.array(positions)
        positions = torch.from_numpy(positions)
        return particle_data, positions
