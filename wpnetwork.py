"""Module for scraping and analysing article networks from a wikipedia category.

This module contains 2 classes: WikiCatScrap and WikiNet.
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import networkx as nx
import numpy as np
import pandas as pd
import statistics as stat
import wptools


class WikiCatScrap:
    
    """Class to scrap Wikipedia articles within one category and the links between them."""
    
    def __init__(self, category, language):
        """Constructor of WikiCatScrap class."""
        self.language = language #fr or en
        self.category = category
        self.info_pages = None
        self.links = None
        
    def get_page_data(self, x): #how to have langue=self.language?
        page = wptools.page(x, lang=self.language, silent=True)
        page = page.get()
        return page.data
    
    def get_summary(self, x):
        return x['exrest']
    
    def get_url(self, x):
        return x['url']
    
    def get_length(self, x):
        return x['length']
    
    def get_links(self, x):
        return x['links']
    
    def get_modification(self, x):
        return x['modified']
    
    def get_wikibase(self, x):
        if 'wikibase' in x:
            return x['wikibase']
        else:
            return np.nan
    
    def get_wikidata_url(self, x):
        if 'wikidata_url' in x:
            return x['wikidata_url']
        else:
            return np.nan
    
    def get_aliases(self, x):
        if 'aliases' in x:
            return x['aliases']
        else:
            return np.nan
    
    def get_cat_pages(self):
        #Get the title of pages of the corpus for the french network
        cat = wptools.category(self.category, lang=self.language)
        membres = cat.get_members()
        self.info_pages = pd.DataFrame.from_dict(membres.data['members'])
        self.info_pages = self.info_pages[['pageid', 'title']]
        #Get infos about the pages
        self.info_pages['content'] = self.info_pages['title'].apply(self.get_page_data)
        #Parse the infos into the DataFrame
        self.info_pages['summary'] = self.info_pages['content'].apply(self.get_summary)
        self.info_pages['url'] = self.info_pages['content'].apply(self.get_url)
        self.info_pages['length'] = self.info_pages['content'].apply(self.get_length)
        self.info_pages['links'] = self.info_pages['content'].apply(self.get_links)
        self.info_pages['modification'] = self.info_pages['content'].apply(self.get_modification)
        self.info_pages['wikibase'] = self.info_pages['content'].apply(self.get_wikibase)
        self.info_pages['wikidata_url'] = self.info_pages['content'].apply(self.get_wikidata_url)
        self.info_pages['aliases'] = self.info_pages['content'].apply(self.get_aliases)
        self.info_pages = self.info_pages.rename(columns={'pageid':'Id'})
        
    def get_links_table(self):
        df = self.info_pages[['Id','title', 'links']]
        lst_col = 'links'
        df_links = pd.DataFrame({col:np.repeat(df[col].values, df[lst_col].str.len())
              for col in df.columns.drop(lst_col)}).assign(**{lst_col:np.concatenate(df[lst_col].values)})[df.columns]
        df_links = df_links.rename(columns={'Id':'Source'})
        df_links = pd.merge(df_links, self.info_pages[['title', 'Id']], left_on='links', right_on='title', how='left').dropna()
        df_links = df_links[['Source', 'Id']].rename(columns={'Id':'Target'})
        df_links['Target'] = df_links['Target'].astype(np.int64)
        self.links = df_links
        
    def process(self):
        self.get_cat_pages()
        self.get_links_table()
    
    #Doesnt work... why?
    def save_as_file(self, name):
        self.wiki_info_pages.to_csv(name + '_pages.csv')
        self.wiki_links.to_csv(name + '_links.csv')


class WikiNet:
    
    """Class to construct a network based on the data scraped by the WikiCatScrap class."""
    
    def __init__(self, pages, links):
        """Constructor of WikiNet class."""
        self.pages = pages
        self.ids = pages['Id']
        self.links = links
        self.path = links[['Source', 'Target']].apply(self.create_path, axis=1)
        self.graph = self.create_graph()
    
    def info(self, full=False):
        """Prints basic infos about the network."""
        print(nx.info(self.graph)[7:])
        print('Average total degree: {0:.4f}'.format(np.mean([d for n, d in self.graph.degree()])))
        if full:
            self.get_connected_components(info=True)
        
    def create_graph(self):
        """Returns self, create a NetworkX graph attribute."""
        graph = nx.DiGraph()
        graph.add_nodes_from(self.ids)
        graph.add_edges_from(self.path)
        return graph
    
    @staticmethod
    def create_path(x):
        """Returns a tuple of the object given."""
        return tuple(x)
    
    def show_graph(self):
        """Plot the class network."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        nx.draw_networkx(self.graph, ax=ax)
    
    @staticmethod
    def select_edges(x, nodes):
        if x['Source'] in nodes and x['Target'] in nodes:
            return x
        else:
            x['Source'] = np.nan
            x['Target'] = np.nan
            return x
    
    def get_subgraph(self, nodes):
        """Returns the subgraph constituted of the nodes selected"""
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        edges = self.links.apply(lambda x: self.select_edges(x, nodes), axis=1)
        edges = edges.dropna().apply(lambda x: self.create_path(x), axis=1)
        graph.add_edges_from(edges)
        return graph
        
    def get_degree_dist(self):
        """Computes degree distibutions of the network and returns la tuple of indegree distribution, 
        outdegree distibution and distribution and their names.
        """
        in_degree = sorted([d for n, d in self.graph.in_degree()], reverse=True)
        out_degree = sorted([d for n, d in self.graph.out_degree()], reverse=True)
        degree = sorted([d for n, d in self.graph.degree()], reverse=True)
        return ([in_degree, out_degree, degree], ['in-degree', 'out-degree', 'total degree'])
    
    @staticmethod
    def hist_degree_dist(degree, title='', figsize=(8,3)):
        """Plots the degee distribution from the DataFrame given as an histogram."""
        fig, ax = plt.subplots(figsize=figsize)
        plt.hist(degree, bins=55)
#         plt.gca().yaxis.set_major_formatter(PercentFormatter(10000, symbol=None))
        plt.title(title + " distribution", fontsize=14)
        plt.ylabel("Number of nodes", fontsize=14)
        plt.xlabel("Degree", fontsize=14)
    
    @staticmethod
    def scatter_degree_dist(degree, title='', figsize=(8,3)):
        """Plots the degee distribution from the DataFrame given as a scatter plot."""
        plt.subplots(figsize=figsize)
        plt.plot(degree, 'yo')
        plt.title("Number of " + title + " per node (sorted by degree)", fontsize=14)
        plt.ylabel("Degree", fontsize=14)
        plt.xlabel("Nodes", fontsize=14)
    
    def log_degree_dist(self, degree, title='', figsize=(8,3)):
        """Plots the degree distribution in log-log scale from the DataFrame given."""
        plt.subplots(figsize=figsize)
        plt.yscale('log')
        plt.xscale('log')
        degree = pd.DataFrame(degree)
        degree = degree.set_index(degree[0]).groupby(level=0).count() / len(self.ids.unique())
        plt.plot(degree, 'ro')
        plt.title(title + " distribution", fontsize=14)
        plt.xlabel('Degree k', fontsize=14)
        plt.ylabel('P(k)', fontsize=14)
        
    def plot_degree_dist(self, kind='all'):
        """Several visualizations of indegree and outdegree distribution of the network."""
        degrees, names = self.get_degree_dist()
        figsize = (8,3)
        if kind == 'all' or kind == 'scatter':
            for deg, name in zip(degrees, names):
                self.scatter_degree_dist(deg, title=name, figsize=(8,3))
        if kind == 'all' or kind == 'hist':
            for deg, name in zip(degrees, names):
                self.hist_degree_dist(deg, title=name, figsize=(8,3))
        if kind == 'all' or kind == 'log':
            for deg, name in zip(degrees, names):
                self.log_degree_dist(deg, title=name, figsize=(8,3))
        plt.show()
    
    @staticmethod
    def get_smallest_components(cc):
        """Returns the nodes of the smallest set of nodes from the ones given in parameter."""
        smallest = []
        for c in cc:
            if len(c) < len(smallest) or smallest == []:
                smallest = c
        return smallest
    
    @staticmethod
    def get_biggest_components(cc):
        """Returns the nodes of the biggest set of nodes from the ones given in parameter."""
        biggest = []
        for c in cc:
            if len(c) > len(biggest) or biggest == []:
                biggest = c
        return biggest
    
    def get_weak_components(self, show=False):
        """Returns a list of lists with nodes for each weak connected component of the class network."""
        weak_components = [list(cc) for cc in nx.weakly_connected_components(self.graph.subgraph(self.ids))]
        if show:
            for cc in weak_components:
                nx.draw_networkx_nodes(self.graph, nx.spring_layout(self.graph), nodelist=cc)
#                 if len(cc) > 1:
#                     nx.draw_networkx_edges(self.graph, nx.spring_layout(self.graph), edgelist=self.path)
                plt.show()
            print("Edges are coming soon.")
        return weak_components 
        
    def get_strong_components(self, show=False):
        """Returns a list of lists with nodes for each strong connected component of the class network."""
        strong_components = [list(cc) for cc in nx.strongly_connected_components(self.graph.subgraph(self.ids))]
        if show:
            for cc in strong_components:
                nx.draw_networkx_nodes(self.graph, nx.spring_layout(self.graph), nodelist=cc)
#                 if len(cc) > 1:
#                     nx.draw_networkx_edges(self.graph, nx.spring_layout(self.graph), edgelist=self.path)
                plt.show()
            print("Edges are coming soon.")
        return strong_components
    
    @staticmethod
    def get_multi_nodes(cc):
        """Selects and returns the nodes of the graphs containing more than one node."""
        multi_cc = []
        for c in cc:
            if len(c) > 1:
                multi_cc.append(c)
        return multi_cc
        
    
    def get_connected_components(self, info=False):
        """Returns nodes of each weak and strong connected component of the class network."""
        weak = self.get_weak_components()
        strong = self.get_strong_components()
        if info:
            print("Number of weak connected components: {}".format(len(weak)))
            print("Number of nodes of the smallest weak cc: {}".format(len(self.get_smallest_components(weak))))
            print("Number of nodes of the biggest weak cc: {}".format(len(self.get_biggest_components(weak))))
            print("Number of weak cc with several nodes: {}".format(len(self.get_multi_nodes(weak))))
            print("Number of strong connected components: {}".format(len(strong)))
            print("Number of nodes of the smallest weak cc: {}".format(len(self.get_smallest_components(strong))))
            print("Number of nodes of the biggest weak cc: {}".format(len(self.get_biggest_components(strong))))
            print("Number of strong cc with several nodes: {}".format(len(self.get_multi_nodes(strong))) )
        return (weak, strong)
    
    def get_avg_path_length(self, weak, strong):
        """Prints the average shortest path length of each weak and strong connected component given in parameter."""
        avg_weak_avg_path = 0
        for cc in weak:
            weak_avg_path = nx.average_shortest_path_length(self.get_subgraph(cc))
            avg_weak_avg_path += weak_avg_path
            print('Average shortest path length in the weak cc with {} nodes: {}'.format(len(cc), weak_avg_path))
        avg_weak_avg_path = avg_weak_avg_path / len(weak)
        print('')
        print('Average shortest path length of the {} weak cc : {}'.format(len(weak), avg_weak_avg_path))
        print('')
        print('--')
        print('')
        avg_strong_avg_path = 0
        for cc in strong:
            strong_avg_path = nx.average_shortest_path_length(self.get_subgraph(cc))
            avg_strong_avg_path += strong_avg_path
            print('Average shortest path length in the strong cc with {} nodes: {}'.format(len(cc), strong_avg_path))
        avg_strong_avg_path = avg_strong_avg_path / len(strong)
        print('')
        print('Average shortest path length of the {} strong cc : {}'.format(len(strong), avg_strong_avg_path))
    
    @staticmethod
    def get_distance_dist(graph):
        all_shortest_path_length = list(nx.shortest_path_length(graph))
        distances = []
        for i in range(len(all_shortest_path_length)):
            for k in all_shortest_path_length[i][1]:
                distances.append(all_shortest_path_length[i][1][k])
        return distances

    @staticmethod
    def plot_distance_dist(graph, figsize=(8,3)):
        """Plots the distance distribution of the class network as an histogram."""
        distances = WikiNet.get_distance_dist(graph)
        print("Distance mean: {}".format(stat.mean(distances)))
        print("Distance variance: {}".format(stat.variance(distances)))
        fig, ax = plt.subplots(figsize=figsize)
        plt.hist(distances, bins=max(distances))
#         plt.gca().yaxis.set_major_formatter(PercentFormatter(10000, symbol=None))
        plt.title("Distance distribution", fontsize=14)
        plt.ylabel("Number of shortest path", fontsize=14)
        plt.xlabel("Distance d", fontsize=14)
        plt.show()
        

    def plot_clustering_per_degree(self):
        df_cluster = pd.DataFrame()
        df_cluster['nodes'] = self.ids
        df_cluster['degree'] = df_cluster['nodes'].apply(lambda x: self.graph.degree[x])
        df_cluster['cluster_coef'] = df_cluster['nodes'].apply(lambda x: nx.clustering(self.graph, x))
        df_plot = df_cluster[['degree', 'cluster_coef']].groupby('degree').mean()
        plt.plot(df_plot, 'o')
        plt.yscale('log')
        plt.xscale('log')
        plt.title('Average local clustering coefficient on the node’s degree', fontsize=14)
        plt.xlabel('Degree k', fontsize=14)
        plt.ylabel('C(k)', fontsize=14)
        

            

                         
        
        
       
    