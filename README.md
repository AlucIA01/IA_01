Conceptual Description of the Code Functionality
Inputs:

The system receives inputs that are numerical values (e.g., two numbers to sum).
Input Neuron Layer:

These neurons receive the input values and start propagating the information to the hidden neurons.
Hidden Neuron Layer:

The hidden neurons process the received signals, applying activations based on the accumulated potential.
Key mechanisms such as synaptic plasticity (STDP), long-term memory, and internal feedback are implemented here.
Synaptic Plasticity (STDP):

The weights of the connections between neurons are dynamically adjusted based on the temporal difference between the activations of pre- and post-synaptic neurons.
Long-Term Memory:

Neurons store a history of activations that influences their future behavior, allowing the network to learn long-term patterns.
Internal Feedback:

Neurons within the hidden layer can feed back into each other, creating internal processing loops that keep the network active and continuously adapting.
Spontaneous Activation:

Neurons can spontaneously activate if they have been inactive for a while, helping the network stay in a state of anticipation.
Output Neuron Layer:

The output layer generates the final result based on the activations of the hidden neurons.
Visualization and Saving:

The network states are saved to a JSON file for later analysis, and graphs are generated to show the network structure and the progress of training.
