Download Link: https://assignmentchef.com/product/solved-ds-ga1008-homework-1-backpropagation
<br>
5/5 - (1 vote)

Backpropagation

The goal of homework 1 is to help you understand the common techniques used in Deep Learning and how to update network parameters by the using backprop- agation algorithm.

Part 1 has two sub-parts, 1.1, 1.2, 1.3 majorly deal with the theory of backprop- agation algorithm whereas 1.4 is to test conceptual knowledge on deep learning. For part 1.2 and 1.3, you need to answer the questions with mathematical equa- tions. You should put all your answers in a PDF file and we will not accept any scanned hand-written answers. It is recommended to use LATEX.

For part 2, you need to program in Python. It requires you to implement your own forward and backward pass without using autograd. You need to submit your mlp.py file for this part.

The due date of homework 1 is 23:55 EST of 09/27. Submit the following files in a zip file your_net_id.zip through NYU Brightspace:

• theory.pdf • mlp.py

The following behaviors will result in penalty of your final score:

<ol>

 <li>5% penalty for submitting your files without using the correct format. (in- cluding naming the zip file, PDF file or python file wrong, or adding extra files in the zip folder, like the testing scripts from part 2).</li>

 <li>20% penalty for late submission within the first 24 hours. We will not accept any late submission after the first 24 hours.</li>

 <li>20% penalty for code submission that cannot be executed using the steps we mentioned in part 2. So please test your code before submit it.1</li>

</ol>

1 Theory (50pt)

To answer questions in this part, you need some basic knowledge of linear algebra and matrix calculus. Also, you need to follow the instructions:

<ol>

 <li>Every vector is treated as column vector.</li>

 <li>You need to use the numerator-layout notation for matrix calculus. Pleaserefer to Wikipedia about the notation.</li>

 <li>You are only allowed to use vector and matrix. You cannot use tensor inany of your answer.</li>

 <li>Missing transpose are considered as wrong answer.</li>

</ol>

1.1 Two-Layer Neural Nets

You are given the following neural net architecture: Linear1 → f → Linear2 → g

where Lineari(x) = W(i)x+b(i) is the i-th affine transformation, and f,g are element-wise nonlinear activation functions. When an input x ∈ Rn is fed to the network, yˆ ∈ RK is obtained as the output.

1.2 Regression Task

We would like to perform regression task. We choose f (·) = (·)+ = ReLU(·) and g to be the identity function. To train this network, we choose MSE loss function lMSE(yˆ, y) = ∥yˆ − y∥2, where y is the target output.

<ol>

 <li>(a)  (1pt) Name and mathematically describe the 5 programming steps you would take to train this model with PyTorch using SGD on a single batch of data.</li>

 <li>(b)  (5pt)Forasingledatapoint(x,y),writedownallinputsandoutputsforfor- ward pass of each layer. You can only use variable x,y,W(1),b(1),W(2),b(2) in your answer. (note that Lineari(x)=W(i)x+b(i)).</li>

</ol>

(c) (8pt) Write down the gradient calculated from the backward pass. You can only use the following variables: x, y,W(1),b(1),W(2),b(2), ∂l , ∂z2 , ∂yˆ in your

∂yˆ ∂z1 ∂z3 answer, where z1,z2,z3,yˆ are the outputs of Linear1,f,Linear2,g.

(d) (3pt) Show us the elements of ∂z2 , ∂yˆ and ∂l (be careful about the dimen-

sionality)?

∂ z 1

∂ z 3 ∂ yˆ

2

1.3 Classification Task

We would like to perform multi-class classification task, so we set both f , g = σ, the logistic sigmoid function σ(z)=. (1+exp(−z))−1.

<ol>

 <li>(a)  (5pt + 8pt + 3pt) If you want to train this network, what do you need to change in the equations of (b), (c) and (d), assuming we are using the same MSE loss function.</li>

 <li>(b)  (1pt) Things are getting better. You realize that not all intermediate hidden activations need to be binary (or soft version of binary). You decide to use f (·) = (·)+ but keep g as σ. Explain why this choice of f can be beneficial for training a (deeper) network.</li>

</ol>

1.4 Conceptual Questions

<ol>

 <li>(a)  (2pt) Why is softmax actually soft(arg)max?</li>

 <li>(b)  (2pt) In what situations, soft(arg)max can become unstable?</li>

 <li>(c)  (2pt) Should we have two consecutive linear layers in a neural network? Why or why not?</li>

 <li>(d)  (4pt)WecoveredvariousactivationfunctionsinclassincludingReLU,Tanh, Sigmoid, and LeakyReLU. Can you give one advantage and disadvantage of each function?</li>

 <li>(e)  (4pt) What are 4 different types of linear transformations? What is the role of linear transformation and non linear transformation in a neural network?</li>

 <li>(f)  (2pt) How should we adjust the learning rate as we increase or decrease the batch size?</li>

</ol>

3

2 Implementation (50pt)

You need to implement the forward pass and backward pass for Linear, ReLU, Sigmoid, MSE loss, and BCE loss in the attached mlp.py file. We provide three example test cases test1.py, test2.py, test3.py. We will test your implemen- tation with other hidden test cases, so please create your own test cases to make sure your implementation is correct.

Recommendation: Go through this Pytorch tutorial to have a thorough under- standing of Tensors.

Extra instructions:

<ol>

 <li>Please use Python version ≥ 3.7 and PyTorch version 1.7.1. We recommend you to use Miniconda the manage your virtual environment.</li>

 <li>We will put your mlp.py file under the same directory of the hidden test scriptsandusethecommandpython hiddenTestScriptName.pytocheck your implementation. So please make sure the file name is mlp.py and it can be executed with the example test scripts we provided.</li>

 <li>You are not allowed to use PyTorch autograd functionality in your imple- mentation.</li>

 <li>Be careful about the dimensionality of the vector and matrix in PyTorch. It is not necessarily follow the the Math you got from part 1.</li>

</ol>