# Linear Regression 

## Variables
|Variable  | Meaning |
|--|--|
| X | Input Variables |
| B | Weights |
| Y | Output |

# Math 🤓

You get the predictions by multiplying the input variables by the appropriate weights. 

<code>X * B = Y</code>

There would also be a matrix that account for the errors, because it is nearly impossible to find coefficients that will fit every single sample perfectly. The NumPy library will minimize this error, so we will ignore the errors matrix for the purposes of this explanation, but do recognize that error minimization is happening in the background.

We are solving for B, so our goal is to isolate it. This means that we only want B on one side of the equation. 

We will want to make sure we have a square matrix on both sides. 

<code>X' * X * B = X' * Y</code>

Square matrices have an inverse. By multiplying it by its inverse in the next step, we get an identity matrix. 

<code>(X' * X)<sup>-1</sup> * X' * X * B = (X' * X)<sup>-1</sup> * X' * Y</code>

We can cancel out the identity matrix to solve for b. 

<code>B = (X' * X)<sup>-1</sup> * X' * Y</code>