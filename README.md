# ece598_project_code

---------------------------------------------------------------------------------
                                    Overall Goal
---------------------------------------------------------------------------------
Reimplement a method to compute the finite-time Lyapunov exponents for a standard
image classifier (Storm 2024).
---------------------------------------------------------------------------------
                                    Current Tasks
---------------------------------------------------------------------------------
1. Circle Demo
    A. Update network output, reference, and the loss function [X]
    
    B. Add weight and threshold paramaters to the network [X]
        Already a thing in the default nn.Tanh setup. I just needed to implement the initialization of the weights andd threshold.

    C. Compute the Jacobian [X]

    D. Compute the first/largest posative eigen value (Finite Time Lyapunov Exponent [FTLE]) [X]

    E. Plot FTLE color gradient over the points [X]

    F. Fix FTLE computation since I did  NOT sort the singular value decomp and for fraction this messes with the lower range of the log ouput. 

2. MNIST Demo

---------------------------------------------------------------------------------
                                    Past Tasks
---------------------------------------------------------------------------------
1. Read Project List
2. Project Survey
3. 1 On 1 Meeting with Prof
4. Read Paper Blind
5. Emailed Authors
6. Setup Repos (Github and Overleaf)
7. Took Notes While Rereading Paper
---------------------------------------------------------------------------------

I'm working on replicating the results from, "Finite-Time Lyapunov Exponents of Deep Neural Networks" from HYSICAL REVIEW LETTERS 132, 057301 (2024).