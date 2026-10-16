# Graph Neural Networks - Group Project AI 2

## Project Overview
This project focuses on the study and application of Graph Neural Networks (GNNs) for analyzing and classifying nodes in complex, real-world networks[cite: 2]. The problem is framed as a binary classification task, aiming to identify network users who have the most complete profiles versus those with incomplete profile details[cite: 3]. 

Furthermore, the study extends beyond technical performance metrics, approaching the problem from a sociological perspective (SSH) to evaluate the algorithmic fairness of the models towards both genders[cite: 2].

## Dataset
* The chosen field of study is a subset of 15,000 users from the Pokec social network (Slovakian App)[cite: 2, 3].
* The network exhibits high heterophily, with a homophily rate of 51.36% and heterophily of 48.64%, while the clustering coefficient is 0.1881[cite: 4].
* Due to server issues during automatic data loading via PyTorch Geometric, the data was manually downloaded from the Stanford SNAP database and extracted using a custom parser[cite: 12, 13].

## Model Architectures & Evaluation
Three different GNN architectures were implemented and compared in this project[cite: 2]:

* **GCN (Graph Convolutional Network):** The model failed to clearly separate the classes (Accuracy ~66%) due to the network's high heterophily, which caused over-smoothing during the aggregation of neighborhood features[cite: 5].
* **H2GCN (Homophily and Heterophily GNN):** Specifically designed for heterophilous networks, this model explicitly separates the central node's information and utilizes 1-hop and 2-hop neighborhoods[cite: 7, 8]. It achieved the highest accuracy, reaching 85%[cite: 8, 12].
* **GAT (Graph Attention Network):** The attention mechanism struggled to find correct attention weights due to extreme heterophily, resulting in the model focusing on noise (underfitting) and recording an accuracy of only 57%[cite: 10].

## Algorithmic Fairness
The models' behavior regarding structural biases based on user gender was examined using Demographic Parity (DP) and Equal Opportunity (EO) metrics[cite: 12]:
* **GCN:** Showed high inequality in Equal Opportunity (EO difference: 0.098), tending to smooth out the unique topological features of minorities and reinforce systemic social biases[cite: 12].
* **GAT:** Exhibited "equality due to failure" (DP difference: 0.011, EO difference: 0.001), as its predictive ability was equally poor across all demographic groups, offering no practical utility[cite: 12].
* **H2GCN:** By protecting the uniqueness of minorities, the model significantly reduced discrimination, improving the EO difference to 0.021 and providing truly equal opportunities to all groups[cite: 12].

## Technologies & Libraries
* **Programming Language:** Python[cite: 2]
* **Machine Learning & GNNs:** PyTorch, PyTorch Geometric (PyG)[cite: 2]
* **Graph Analysis:** NetworkX[cite: 2]
* **Data Processing & Visualization:** Sklearn, Pandas, Seaborn[cite: 2]
* **Development Environment:** Google Colab[cite: 2]

## Team Members
* Dimitrios Filopoulos (1101130)
* Nektarios Kossyfas (1100902)
* Vasileios Roussis (1104685)
* Theodoros Latsis (1104676)
