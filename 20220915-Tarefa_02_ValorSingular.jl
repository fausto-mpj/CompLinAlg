##
using Chain
using DoubleFloats
using LinearAlgebra
using Plots
using MatrixDepot

##
A = matrixdepot("cauchy", 3)
alvo = 1.0 / norm(inv(A), 2)

##
U, Σ, V = svd(A)
σₙ = minimum(Σ)
Σₖ = zeros(Float64, size(A))
setindex!(Σₖ, -σₙ, length(Σₖ))

##
Eₖ = U * Σₖ * transpose(V)
Aₑ = A + Eₖ

##
inv(Aₑ)

##
det(Aₑ)

##
svd(Aₑ)
