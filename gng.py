#!/usr/bin/env python
# coding: utf-8

# In[12]:


# coding: utf-8

import numpy as np
from scipy import spatial
import networkx as nx
import matplotlib.pyplot as plt
from sklearn import decomposition
'''
Simple implementation of the Growing Neural Gas algorithm, based on:
A Growing Neural Gas Network Learns Topologies. B. Fritzke, Advances in Neural
Information Processing Systems 7, 1995.
'''

import chart_studio
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from gensim.summarization.summarizer import summarize
#https://radimrehurek.com/gensim/summarization/summariser.html


class GrowingNeuralGas:

    def __init__(self, input_data, labels):
        self.network = None
        self.data = input_data
        self.labels = labels
        self.units_created = 0
        plt.style.use('ggplot')
        
    def get_distance(self, a, b):
        return spatial.distance.cosine(a, b)

    def find_nearest_units(self, observation):
        distance = []
        for u, attributes in self.network.nodes(data=True):
            vector = attributes['vector']
            dist = self.get_distance(vector, observation)
            distance.append((u, dist))
        distance.sort(key=lambda x: x[1])
        ranking = [u for u, dist in distance]
        return ranking

    def get_summary(self, node):
        distance = []
        for i in range(len(self.data)):
            dist = self.get_distance(node, self.data[i])
            distance.append((i, dist))
        distance.sort(key=lambda x: x[1])
        
        #window = len(self.data) / self.network.number_of_nodes() * 0.5
        window = 20
        articles = ""
        for (i, dist) in distance[:window]:
            articles += self.labels[i] + "\n"
        summarized_text = summarize(articles, ratio=0.12, split=False)
        print("#####################")
        print(summarized_text)
        print("#####################")
        
        return summarized_text

    
    def prune_connections(self, a_max, e_max):
        for u, v, attributes in self.network.edges(data=True):
            if attributes['age'] > a_max:
                self.network.remove_edge(u, v)
        for node in self.network.nodes():
            degree = self.network.degree(node)
            if degree == 0:
                self.network.remove_node(node)
            if degree > e_max:
                max_dist = 0.0
                edge = None
                for u, v, attributes in self.network.edges(node, data=True):
                    dist = self.get_distance(u,v)
                    if max_dist <= dist:
                        max_dist = dist
                        edge = (u, v)
                self.network.remove_edge(edge[0], edge[1])
                

    def fit_network(self, e_b, e_n, a_max, l, a, d, e_max=5, n_max=100, initial_nodes_num=20, passes=1, plot_evolution=False, data_shuffle=False, animation_interval=30):
        # logging variables
        accumulated_local_error = []
        global_error = []
        network_order = []
        network_size = []
        total_units = []
        self.units_created = 0
        
        # 0. start with two units a and b at random position w_a and w_b
        # 0. create initial nodes at random position
        self.network = nx.Graph()
        for x in range(initial_nodes_num):
            w = [np.random.uniform(-0.1, 0.1) for _ in range(np.shape(self.data)[1])]
            self.network.add_node(self.units_created, vector=w, error=0)
            self.units_created += 1
        
        # 1. iterate through the data
        sequence = 0
        observations = self.data[:]
        for p in range(passes):
            print('   Pass #%d' % (p + 1))
            if data_shuffle:
                np.random.shuffle(self.data)
                
            steps = 0
            
            np.flip(observations, axis=0)
            for observation in observations:
                # 2. find the nearest unit s_1 and the second nearest unit s_2
                nearest_units = self.find_nearest_units(observation)
                s_1 = nearest_units[0]
                s_2 = nearest_units[1]
                # 3. increment the age of all edges emanating from s_1
                for u, v, attributes in self.network.edges_iter(data=True, nbunch=[s_1]):
                    self.network.add_edge(u, v, age=attributes['age']+1)
                # 4. add the cosine distance between the observation and the nearest unit in input space
                self.network.node[s_1]['error'] += self.get_distance(observation, self.network.node[s_1]['vector'])
                # 5 .move s_1 and its direct topological neighbors towards the observation by the fractions
                #    e_b and e_n, respectively, of the total distance
                update_w_s_1 = e_b * (np.subtract(observation, self.network.node[s_1]['vector']))
                self.network.node[s_1]['vector'] = np.add(self.network.node[s_1]['vector'], update_w_s_1)
                
                for neighbor in self.network.neighbors(s_1):
                    update_w_s_n = e_n * (np.subtract(observation, self.network.node[neighbor]['vector']))
                    self.network.node[neighbor]['vector'] = np.add(self.network.node[neighbor]['vector'], update_w_s_n)
                # 6. if s_1 and s_2 are connected by an edge, set the age of this edge to zero
                #    if such an edge doesn't exist, create it
                self.network.add_edge(s_1, s_2, age=0)
                # 7. remove edges with an age larger than a_max
                #    if this results in units having no emanating edges, remove them as well
                self.prune_connections(a_max, e_max)
                # 8. if the number of steps so far is an integer multiple of parameter l, insert a new unit
                steps += 1
                
                if steps % animation_interval == 0:
                    if plot_evolution:
                        self.plot_network('visualization/sequence/' + str(sequence) + '.png')
                    sequence += 1
                    
                if steps % l == 0 and self.network.number_of_nodes() <  n_max:
                    # 8.a determine the unit q with the maximum accumulated error
                    q = 0
                    error_max = 0
                    for u in self.network.nodes_iter():
                        if self.network.node[u]['error'] > error_max:
                            error_max = self.network.node[u]['error']
                            q = u
                    # 8.b insert a new unit r halfway between q and its neighbor f with the largest error variable
                    f = -1
                    largest_error = -1
                    for u in self.network.neighbors(q):
                        if self.network.node[u]['error'] > largest_error:
                            largest_error = self.network.node[u]['error']
                            f = u
                    w_r = 0.5 * (np.add(self.network.node[q]['vector'], self.network.node[f]['vector']))
                    r = self.units_created
                    self.units_created += 1
                    # 8.c insert edges connecting the new unit r with q and f
                    #     remove the original edge between q and f
                    self.network.add_node(r, vector=w_r, error=0)
                    self.network.add_edge(r, q, age=0)
                    self.network.add_edge(r, f, age=0)
                    self.network.remove_edge(q, f)
                    # 8.d decrease the error variables of q and f by multiplying them with a
                    #     initialize the error variable of r with the new value of the error variable of q
                    self.network.node[q]['error'] *= a
                    self.network.node[f]['error'] *= a
                    self.network.node[r]['error'] = self.network.node[q]['error']
                        
                # 9. decrease all error variables by multiplying them with a constant d
                error = 0
                for u in self.network.nodes_iter():
                    error += self.network.node[u]['error']
                accumulated_local_error.append(error)
                network_order.append(self.network.order())
                network_size.append(self.network.size())
                total_units.append(self.units_created)
                for u in self.network.nodes_iter():
                    self.network.node[u]['error'] *= d
                    if self.network.degree(nbunch=[u]) == 0:
                        print(u)
            global_error.append(self.compute_global_error())
        plt.clf()
        plt.title('Accumulated local error')
        plt.xlabel('iterations')
        plt.plot(range(len(accumulated_local_error)), accumulated_local_error)
        plt.savefig('visualization/accumulated_local_error.png')
        plt.clf()
        plt.title('Global error')
        plt.xlabel('passes')
        plt.plot(range(len(global_error)), global_error)
        plt.savefig('visualization/global_error.png')
        plt.clf()
        plt.title('Neural network properties')
        plt.plot(range(len(network_order)), network_order, label='Network order')
        plt.plot(range(len(network_size)), network_size, label='Network size')
        plt.legend()
        plt.savefig('visualization/network_properties.png')
        
        
        node_vetors = [self.network.node[u]['vector'] for u in self.network.nodes_iter()]
        nodes_num = len(node_vetors)
        pca = decomposition.PCA(n_components=2)
        
        y = []
        y.extend(node_vetors)
        y.extend(self.data)
        
        y_pos = pca.fit_transform(y)
        
        nodes_pos = y[:nodes_num]
        data_pos = y[nodes_num:]
        
        labels = [self.get_summary(vector) for vector in node_vetors]

        init_notebook_mode(connected=True)

        data = [
            go.Scatter(
                x=[i[0] for i in data_pos],
                y=[i[1] for i in data_pos],
                mode='markers',
                text=[i for i in self.labels],
                marker=dict(
                    size=16,
                    color = [len(i) for i in self.labels], #set color equal to a variable
                    opacity= 0.8,
                    colorscale='Viridis',
                    showscale=False
                )
            ),
            go.Scatter(
                x=[i[0] for i in nodes_pos],
                y=[i[1] for i in nodes_pos],
                mode='markers',
                text=labels,
                marker=dict(
                    size=35,
                    color = [len(i) for i in labels], #set color equal to a variable
                    opacity= 0.8,
                    colorscale='Viridis',
                    showscale=False
                )
            )
        ]
        layout = go.Layout()
        layout = dict(
                      yaxis = dict(zeroline = False),
                      xaxis = dict(zeroline = False)
                     )
        fig = go.Figure(data=data, layout=layout)
        file = plot(fig, filename='visualization/result.html')

    def plot_network(self, file_path):
        plt.clf()
        plt.scatter(self.data[:, 0], self.data[:, 1])
        node_pos = {}
        #node_labels = {}
        for u in self.network.nodes_iter():
            vector = self.network.node[u]['vector']
            node_pos[u] = (vector[0], vector[1])
            #node_labels[u] = str(self.find_nearest_data(vector)[0])
        
        nx.draw(self.network, pos=node_pos) #labels=node_labels
        #nx.draw_networkx_labels(self.network, pos=node_pos, labels=node_labels, font_size=8, font_color='k', alpha=0.8)
        plt.draw()
        plt.savefig(file_path)

    def number_of_clusters(self):
        return nx.number_connected_components(self.network)

    def cluster_data(self):
        unit_to_cluster = np.zeros(self.units_created)
        cluster = 0
        for c in nx.connected_components(self.network):
            for unit in c:
                unit_to_cluster[unit] = cluster
            cluster += 1
        clustered_data = []
        for observation in self.data:
            nearest_units = self.find_nearest_units(observation)
            s = nearest_units[0]
            clustered_data.append((observation, unit_to_cluster[s]))
        return clustered_data

    def reduce_dimension(self, clustered_data):
        transformed_clustered_data = []
        svd = decomposition.PCA(n_components=2)
        transformed_observations = svd.fit_transform(self.data)
        for i in range(len(clustered_data)):
            transformed_clustered_data.append((transformed_observations[i], clustered_data[i][1]))
        return transformed_clustered_data

    def plot_clusters(self, clustered_data):
        number_of_clusters = nx.number_connected_components(self.network)
        plt.clf()
        plt.title('Cluster affectation')
        color = ['r', 'b', 'g', 'k', 'm', 'r', 'b', 'g', 'k', 'm']
        for i in range(number_of_clusters):
            observations = [observation for observation, s in clustered_data if s == i]
            if len(observations) > 0:
                observations = np.array(observations)
                plt.scatter(observations[:, 0], observations[:, 1], color=color[i], label='cluster #'+str(i))
        plt.legend()
        plt.savefig('visualization/clusters.png')

    def compute_global_error(self):
        global_error = 0
        for observation in self.data:
            nearest_units = self.find_nearest_units(observation)
            s_1 = nearest_units[0]
            global_error += spatial.distance.euclidean(observation, self.network.node[s_1]['vector'])**2
        return global_error
