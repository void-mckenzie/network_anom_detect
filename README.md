# Network Intrusion Detection using AutoEncoders
AutoEncoder assissted anomaly detection on KDD+ Dataset

Network security is one of the most critical fields of computer science. With the advent of IoT technologies and peer-to-peer networks, the significance of mitigating security threats has never been higher. Network Intrusion Detection Systems are used to monitor the traffic in a network to detect any malicious or anomalous behavior. Anomalous behaviour includes different types of attacks such as Denial of Service (DoS), Probe, Remote-to-Local and User-to-Root. If an attack/anomaly is detected, custom alerts can be sent to the desired personals.

In this paper, we will be exploring the effectiveness of various types of Autoencoders in detecting network intrusions. Artificial Neural Networks can parse through vast amounts of data to detect various types of anomalies and classify them accordingly. An autoencoder is a type of artificial neural network which can learn both linear and non-linear representations of the data, and use the learned representations to reconstruct the original data. These hidden representations are different from the ones attained by Principal Component Analysis due to the presence of non-linear activation functions in the network. Reconstruction error (the measure of difference between the original input and the reconstructed input) is generally used to detect anomalies if the autoencoder is trained on normal network data.

Here, we used 4 different autoencoders on the NLS-KDD dataset to detect attacks in the network. With just reconstruction error, we were able to achieve a highest accuracy of 89.34% by using a Sparse Deep Denoising Autoencoder.

