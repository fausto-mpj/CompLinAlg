### A Pluto.jl notebook ###
# v0.19.14

using Markdown
using InteractiveUtils

# ╔═╡ 88948702-62b7-11ed-0862-a94108822a9a
# Pacotes
begin
	using Chain
	using Latexify
	using LaTeXStrings
	using LinearAlgebra
	using Random
	using Gadfly
end

# ╔═╡ f17f1b3c-0cb5-476e-a756-fc634ed53ec0
md"""
# Autovalores e Autovetores

**Matriz Exemplo 1:** Definimos como matriz $A$ a seguinte matriz abaixo:
"""

# ╔═╡ cd5be5c8-ec66-49cf-a35f-581f6945e2de
begin
	A = [50.0 7.0 9.0; 7.0 -80.0 -6.0; 9.0 -6.0 10.0]
	latexify(A)
end

# ╔═╡ 9fcefb8c-dd1b-4d4f-805b-4860ceda0499
md"""
Para fins de comparação, seguem os autovalores e autovetores de $A$ obtidos pela função padrão no *Julia* para a decomposição espectral.
"""

# ╔═╡ 3274165a-6d9e-4e88-a22e-b5271ba90c09
eigen(A)

# ╔═╡ 73be5700-36ec-49e2-af59-d3e6ec205fbd
md"""
## Método das Potências

### Suposições:

* A matriz $A \in \mathbb{K}^{n \times n}$ é diagonalizável, ou seja, os autovetores da matriz formam uma base;


* Os autovalores de $A$ satisfazem a relação

$$\vert \lambda_{1} \vert > \vert \lambda_{2} \vert \geq \vert \lambda_{3} \vert \geq \ldots \geq \vert \lambda_{n} \vert.$$

"""

# ╔═╡ 7d9eaf4e-bd8f-4a32-8faf-538aa649fb0e
md"""
### Algoritmo:
"""

# ╔═╡ bd9c73d9-7e66-4666-9944-3d0f02f841d2
function metodo_potencia(A::Matrix; ϵ::Float64 = 10^-8, maxiter::Int64 = 10^6)
	# Gerar um vetor aleatório unitário `w`
	w = @chain begin
		rand(Float64, size(A, 2))
		_ / norm(_, 2)
	end
	λ₀ = Inf
	iter = 0
	while true
		iter += 1
		# Gerar um novo vetor `z` a partir de `A * w` tal que `A * w = λ * w = z`
		z = A * w
		# Multiplicar por `w^{T}` tal que `λ = λ w^{T} w = w^{T} * z`
		λ = transpose(w) * z
		# Normalizar o novo `w` para mantê-lo como unitário
		w = z/norm(z, 2)
		if abs(λ - λ₀) < ϵ
			return(valor = λ, resultado = "Convergiu em $(iter) iterações")
		elseif iter == maxiter
			return(valor = λ, resultado = "Não convergiu em $(iter) iterações")
		else
			λ₀ = λ
		end
	end
end

# ╔═╡ 4e964318-5a08-4e4e-9660-b75c3048b008
λ₁ = metodo_potencia(A)

# ╔═╡ 11d2b341-ee95-4d34-939f-8f7db8d2e8d0
md"""
## Método da Iteração Inversa

### Motivação:

* Se $A$ é não-singular e $\lambda_{i}$ é autovalor de $A$, então $1/\lambda_{i}$ é autovalor de $A^{-1}$;

### Suposições:

* A matriz $A \in \mathbb{K}^{n \times n}$ é diagonalizável, ou seja, os autovetores da matriz formam uma base;


* Os autovalores de $A^{-1}$ satisfazem a relação

$$\vert 1 / \lambda_{n} \vert > \vert 1 / \lambda_{n-1} \vert \geq \vert 1 / \lambda_{n - 2} \vert \geq \ldots \geq \vert 1 / \lambda_{2} \vert > \vert 1 / \lambda_{1} \vert.$$

"""

# ╔═╡ fa1eea07-e0e8-48c6-9c9d-04f6ff5fff8f
md"""
### Algoritmo:
"""

# ╔═╡ b142b019-9deb-425a-b906-78d2108cd6e4
function metodo_potencia_inversa(A::Matrix; ϵ::Float64 = 10^-8, maxiter::Int64 = 10^6)
	w = @chain begin
		rand(Float64, size(A, 2))
		_ / norm(_, 2)
	end
	# Note que se `A = L * U`, então `A^{-1} = U^{-1} * L^{-1}`.
	L,U = lu(A)
	L⁻¹ = inv(L)
	U⁻¹ = inv(U)
	# Note que se `λ ≠ 0` é autovalor de `A`, então `β = 1/λ` é autovalor de `A^{-1}`.
	β₀ = Inf
	iter = 0
	while true
		iter += 1
		# Abaixo temos que atribuir ao `z` o valor `A^{-1} * w` usando a LU
		z = U⁻¹ * L⁻¹ * w
		β = transpose(w) * z
		w = z/norm(z, 2)
		if abs(β - β₀) < ϵ
			return(valor = 1/β, resultado = "Convergiu em $(iter) iterações")
		elseif iter == maxiter
			return(valor = 1/β, resultado = "Não convergiu em $(iter) iterações")
		else
			β₀ = β
		end
	end
end

# ╔═╡ 59e3c1ba-34d7-4fc3-ac8c-317a9c55b1c1
λₙ = metodo_potencia_inversa(A)

# ╔═╡ eb7bbeae-0401-4702-b8ce-e24c1ea28e8f
md"""
## Método da Potência com *Shift*

### Motivação:

* Se $\lambda_{i}$ é autovalor de $A$, então a diferença $\delta_{i} = (\lambda_{i} - \alpha)$ é autovalor de $B = (A - \alpha I)$.

### Suposições:

* A matriz $A \in \mathbb{K}^{n \times n}$ é diagonalizável, ou seja, os autovetores da matriz formam uma base;

* Os autovalores de $A$ satisfazem a relação

$$\vert \lambda_{1} \vert > \vert \lambda_{2} \vert \geq \vert \lambda_{3} \vert \geq \ldots \geq \vert \lambda_{n} \vert.$$

"""

# ╔═╡ 6b464cd7-a759-4507-8f82-91737c8ee7f5
md"""

**Matriz Exemplo 2:** Definimos como matriz $B = (A - \alpha I)$, com $\alpha =$ $(metodo_potencia(A).valor), a seguinte matriz abaixo:

"""

# ╔═╡ 898d5516-6ab7-46d5-9d7e-628096e8b8cf
begin
	B = A - I * metodo_potencia(A).valor
	latexify(B)
end

# ╔═╡ 93225e28-7277-4906-a49a-16c6f8e6cdaa
δ₁ = metodo_potencia(A - I * metodo_potencia(A).valor)

# ╔═╡ ead9093d-07fb-4c43-8985-5a114b97071c
md"""
Como $\delta_{1}$ é autovalor de $B$ e $\alpha = \lambda_{1}$, então sabemos que a soma $(\delta_{1} + \lambda_{1})$ é autovalor de $A$.
"""

# ╔═╡ e4e6d0c7-4ed8-4a33-a064-7632e0893b34
metodo_potencia(A - I * metodo_potencia(A).valor).valor + metodo_potencia(A).valor

# ╔═╡ 67eec47c-cf99-430e-a618-0df548200a82
md"""
### Algoritmo:
"""

# ╔═╡ c2afceac-1fe7-40bc-9e29-2127edc6947f
function metodo_potencia_shift(A::Matrix, α::Float64; ϵ::Float64 = 10^-8, maxiter::Int64 = 10^6)
	w = @chain begin
		rand(Float64, size(A, 2))
		_ / norm(_, 2)
	end
	B = A - α * I
	δ₀ = Inf
	iter = 0
	while true
		iter += 1
		z = B * w
		δ = transpose(w) * z
		w = z/norm(z, 2)
		if abs(δ - δ₀) < ϵ
			return(valor = δ + α, resultado = "Convergiu em $(iter) iterações")
		elseif iter == maxiter
			return(valor = δ + α, resultado = "Não convergiu em $(iter) iterações")
		else
			δ₀ = δ
		end
	end
end

# ╔═╡ a6de036e-7d32-47ca-b0ab-a563e976a6d5
γ = metodo_potencia_shift(A, metodo_potencia(A).valor)

# ╔═╡ 947e5d14-ce3e-4141-bb96-e3d21d15f35d
md"""
## Método da Iteração Inversa com *Shift*

### Motivação:

* Se $A$ é não-singular e $\lambda_{i}$ é autovalor de $A$, então $1/\lambda_{i}$ é autovalor de $A^{-1}$;


* Se $\lambda_{i}$ é autovalor de $A$, então a diferença $\delta_{i} = (\lambda_{i} - \alpha)$ é autovalor de $B = (A - \alpha I)$.


* Observe que o menor número de iterações para a convergência ocorreu no Método da Iteração Inversa. Isso decorre do fato da convergência do método da potência se dar com taxa proporcional à razão $\vert \lambda_{1} \vert / \vert \lambda_{2} \vert$. No entanto, se escolhermos um *shift* $\alpha$ que está muito próximo de algum autovalor $\lambda_{i}$, então temos que $\lambda_{i} - \alpha \approx 0$ e, portanto, $1 / (\vert \lambda_{i} - \alpha \vert) \gg 1 / (\vert \lambda_{j} - \alpha \vert)$, $\forall j \neq i$. Assim, a convergência será mais rápida.


* Seja $B^{-1} = (A - \alpha I)^{-1}$ e $\beta$ autovalor de $B^{-1}$. Então temos que $1 / \beta$ é autrovalor de $B$ e a soma $(\alpha + 1/ \beta)$ é autovalor de $A$.


* No processo com *shift*, na ausência de maiores informações, podemos usar como valor de $\alpha$ os valores da diagonal de $A$.

### Suposições:

* A matriz $A \in \mathbb{K}^{n \times n}$ é diagonalizável, ou seja, os autovetores da matriz formam uma base;


* Os autovalores de $A^{-1}$ satisfazem a relação

$$\vert 1 / \lambda_{n} \vert > \vert 1 / \lambda_{n-1} \vert \geq \vert 1 / \lambda_{n - 2} \vert \geq \ldots \geq \vert 1 / \lambda_{2} \vert > \vert 1 / \lambda_{1} \vert.$$

"""

# ╔═╡ 55b3af7e-cf75-4ef7-a5aa-f038ef61b0ff
md"""
### Algoritmo:
"""

# ╔═╡ d323edc9-0f07-4b26-80af-37ffd417deaf
function metodo_potencia_inversa_shift(A::Matrix, α::Float64; ϵ::Float64 = 10^-8, maxiter::Int64 = 10^6)
	w = @chain begin
		rand(Float64, size(A, 2))
		_ / norm(_, 2)
	end
	B = A - α * I
	# Note que se `B = L * U`, então `B^{-1} = U^{-1} * L^{-1}`.
	L,U = lu(B)
	L⁻¹ = inv(L)
	U⁻¹ = inv(U)
	# Note que se `λ ≠ 0` é autovalor de `A`, então `β = 1/λ` é autovalor de `A^{-1}`.
	β₀ = Inf
	iter = 0
	while true
		iter += 1
		# Abaixo temos que atribuir ao `z` o valor `B^{-1} * w` usando a LU
		z = U⁻¹ * L⁻¹ * w
		β = transpose(w) * z
		w = z/norm(z, 2)
		if abs(β - β₀) < ϵ
			return(valor = 1/β + α, resultado = "Convergiu em $(iter) iterações")
		elseif iter == maxiter
			return(valor = 1/β + α, resultado = "Não convergiu em $(iter) iterações")
		else
			β₀ = β
		end
	end
end

# ╔═╡ 3bf0092d-d8b1-4610-af55-c93bb42b307b
metodo_potencia_inversa_shift(A, A[1,1])

# ╔═╡ e21a239d-176d-4da7-a71e-25e9f4388605
metodo_potencia_inversa_shift(A, A[2,2])

# ╔═╡ f0fe0a76-6aa2-45dd-b4d0-70dfd1618516
metodo_potencia_inversa_shift(A, A[3,3])

# ╔═╡ 04d987cc-bbbd-4a1b-a48b-25a9ce107af9
metodo_potencia_inversa_shift(A, metodo_potencia_inversa(A).valor)

# ╔═╡ d02eb301-d769-438d-b758-35e2a75499dc
metodo_potencia_inversa_shift(A, metodo_potencia(A).valor)

# ╔═╡ f054bc30-ecae-444e-ad28-573991170f33
metodo_potencia_inversa_shift(A, metodo_potencia_shift(A, metodo_potencia(A).valor).valor)

# ╔═╡ 57f5909b-3e54-42dd-a2b8-d6f2f8591f23
md"""
Note que nem sempre o *shift* escolhido resultou na convergência para os autovalores para os critérios de parada escolhidos (precisão e número máximo de iterações).

Além disso, o método da iteração inversa com *shift* é o **único** dentre os vistos acima capaz de detectar autovalores que não sejam extremos, ou seja, com maior ou menor valor absoluto.
"""

# ╔═╡ bb6661ad-b93e-4cbc-a903-ec7149a74b9b
md"""
## Método da Iteração Inversa com *Shift* de Rayleigh

### Motivação:

* Se $A$ é não-singular e $\lambda_{i}$ é autovalor de $A$, então $1/\lambda_{i}$ é autovalor de $A^{-1}$;


* Se $\lambda_{i}$ é autovalor de $A$, então a diferença $\delta_{i} = (\lambda_{i} - \alpha)$ é autovalor de $B = (A - \alpha I)$.


* Observe que o menor número de iterações para a convergência ocorreu no Método da Iteração Inversa. Isso decorre do fato da convergência do método da potência se dar com taxa proporcional à razão $\vert \lambda_{1} \vert / \vert \lambda_{2} \vert$. No entanto, se escolhermos um *shift* $\alpha$ que está muito próximo de algum autovalor $\lambda_{i}$, então temos que $\lambda_{i} - \alpha \approx 0$ e, portanto, $1 / (\vert \lambda_{i} - \alpha \vert) \gg 1 / (\vert \lambda_{j} - \alpha \vert)$, $\forall j \neq i$. Assim, a convergência será mais rápida.


* Seja $B^{-1} = (A - \alpha I)^{-1}$ e $\beta$ autovalor de $B^{-1}$. Então temos que $1 / \beta$ é autrovalor de $B$ e a soma $(\alpha + 1/ \beta)$ é autovalor de $A$.


* Ao invés de escolhermos um *shift* arbitrário, altera-se o *shift* em cada iteração de modo que fique mais próximo de um autovalor. Desta forma, amentamos a chance de convergência.


* O valor $\rho$ solução de $Aw = \rho w$, para um $w$ autovetor de $A$, é dito *quociente de Rayleigh* e obtido por mínimos quadrados pela equação:

$$\rho = \frac{w^{H} A w}{w^{H} w}.$$

### Suposições:

* A matriz $A \in \mathbb{K}^{n \times n}$ é diagonalizável, ou seja, os autovetores da matriz formam uma base;


* Os autovalores de $A^{-1}$ satisfazem a relação

$$\vert 1 / \lambda_{n} \vert > \vert 1 / \lambda_{n-1} \vert \geq \vert 1 / \lambda_{n - 2} \vert \geq \ldots \geq \vert 1 / \lambda_{2} \vert > \vert 1 / \lambda_{1} \vert.$$

"""

# ╔═╡ b0974dd6-3617-45c3-a258-7c6113461f90
md"""
### Algoritmo:
"""

# ╔═╡ c8709478-f17f-4674-bd24-15a22672a7df
function metodo_shift_rayleigh(A::Matrix; ϵ::Float64 = 10^-8, maxiter::Int64 = 10^6)
	w = @chain begin
		rand(Float64, size(A, 2))
		_ / norm(_, 2)
	end
	ρ₀ = transpose(w) * A * w
	iter = 0
	while true
		iter += 1
		z = (A - ρ₀ * I) \ w
		w = z/norm(z, 2)
		ρ = transpose(w) * A * w		
		if abs(ρ - ρ₀) < ϵ
			return(valor = ρ, resultado = "Convergiu em $(iter) iterações")
		elseif iter == maxiter
			return(valor = ρ, resultado = "Não convergiu em $(iter) iterações")
		else
			ρ₀ = ρ
		end
	end
end

# ╔═╡ 5b5709c2-ba16-4569-84ca-d04524102954
metodo_shift_rayleigh(A)

# ╔═╡ e4ce0e06-6409-4e6e-bc21-e68488f45d0f
metodo_shift_rayleigh(A)

# ╔═╡ 8a142039-b13d-42e1-a68e-ec415d37d16a
metodo_shift_rayleigh(A)

# ╔═╡ ad2a2fd8-d719-496e-be37-e1a7e85992df
md"""
Neste método não há garantia de convergência para um determinado autovetor e seu autovalor correspondente. Ou seja, o método não alcança somente os autovalores extremos, como, por exemplo, o método das potências (que leva ao maior autovalor em módulo) ou o método da iteração inversa (que leva ao menor autovalor em módulo).

A vantagem deste método é que quando há convergência, ela ocorre em poucas iterações quando comparado com os métodos anteriores. Em particular, no caso de matrizes simétricas, a taxa de convergência é cúbica.
"""

# ╔═╡ 8b5f7ffa-a04a-4fbd-a560-51e45b975c35
md"""
## Método da Iteração QR

### Motivação:

* Se $A$ for mal-condicionada, então os procedimentos para obter os autovalores podem ser imprecisos devido aos erros de arredondamento. O emprego de transformações ortogonais é então útil, porque produz transformações mais numericamente estáveis.


* As matrizes $A$ e $B$ são **similares** se existe uma matriz $P$ tal que $B = P^{-1} A P$.


* Se $A$ e $B$ são semelhantes, então possuem os mesmos autovalores.


* Uma matriz $A$ é **simples** se, e somente se, $A$ é similar a alguma matriz diagonal $D$.


* As matrizes $A$ e $B$ são **unitariamente semelhantes** se, e somente se, $B = U^{H} A U$, onde $U$ é uma matriz unitária.


* Se $A$ e $B$ são unitariamente semelhantes, então possuem os mesmos autovalores.


* **Teorema de Shur:** Toda matriz $A \in \mathbb{C}^{n \times n}$ é unitariamente semelhante a alguma matriz triangular superior (ou inferior) $T$, tal que $T = Q^{H} A Q$, com $Q$ matriz unitária.


* **Teorema de Shur (em $\mathbb{R}$):** Toda matriz $A \in \mathbb{R}^{n \times n}$ é ortogonalmente semelhante a alguma matriz bloco-triangular superior (ou inferior) $T$, tal que $T = Q^{T} A Q$, com $Q$ matriz ortogonal.


### Suposições:


* Seja $Q = [Q_{p} \ Q_{q}]$ e $T$ matriz bloco-triangular superior, onde $T_{1,1}$ tem dimensão $p \times p$. Se $\vert \lambda_{p} \vert > \vert \lambda_{p+1} \vert$, então o subespaço $Im(Q_{p})$ é um **espaço invariante dominante** e as colunas de $Q_{k}$ vão se tornando ortogonais em relação às demais quando $k \rightarrow \infty$.


* Seja $T_{k-1} = Q^{H}_{k-1} A Q_{k-1}$, então $A Q_{k-1} = Q_{k} R_{k}$, pela decomposição $QR$, e $T_{k-1} = Q^{H}_{k-1} Q_{k} R_{k}$. Assim, para $k \rightarrow \infty$, temos que $Q^{H}_{k-1} Q_{k} \rightarrow I$ e $R_{k} \rightarrow T$.


* Assim: $T_{k} = Q^{H}_{k} A Q_{k} = Q^{H}_{k} A Q_{k-1} Q^{H}_{k-1} Q_{k} = Q^{H}_{k} Q_{k} R_{k} Q^{H}_{k-1} Q_{k} = R_{k} Q^{H}_{k-1} Q_{k}$. Segue que $Q^{H}_{k} A = R_{k} Q^{H}_{k-1}$.


* Seja agora $A^{(0)} = Q R$. Então $A^{(1)} = Q^{T} A^{(0)} Q = R Q$, onde $A^{(0)}$ e $A^{(1)}$ têm os mesmos autovalores e podemos usar $A$ para iterativamente encontrar os seus autovalores e autovetores.

"""

# ╔═╡ 47db7ece-5379-4323-8357-66353af0e747
md"""
### Algoritmo:
"""

# ╔═╡ ba5264a5-565a-4ae7-afa7-7800c6d556ea
function metodo_qr(Aₖ::Matrix; ϵ::Float64 = 10^-8, maxiter::Int64 = 10^6)
	Qₖ, Rₖ = qr(Aₖ)
	iter = 0
	while true
		iter += 1
		Aₖ = Rₖ * Qₖ
		Qₖ₊₁, Rₖ₊₁ = qr(Aₖ)
		if norm(transpose(Qₖ) * Qₖ₊₁ - I, 2) < ϵ
			return(valor = Aₖ, resultado = "Convergiu em $(iter) iterações")
		elseif iter == maxiter
			return(valor = Aₖ, resultado = "Não convergiu em $(iter) iterações")
		else
			Qₖ, Rₖ = Qₖ₊₁, Rₖ₊₁
		end
	end
end

# ╔═╡ 38869197-dd65-4c35-a89f-0a535a08e63c
metodo_qr(A)

# ╔═╡ 4f85f994-4fc9-4e63-a56a-ed5da19328cb
md"""
## Método da Iteração QR-Hessenberg com *Shift*

### Motivação:

* A decomposição $QR$ é um procedimento computacionalmente custoso. No entanto, em matrizes Hessenberg o procedimento é bem mais barato, sendo necessário somente $n-1$ rotações de Givens.


* Seja $Q^{T}_{h}$ uma matriz de Householder que anula os elementos abaixo da subdiagonal de $A$. Então $Q^{T}_{h} A Q_{h} = H$, com $H$ matriz Hessenberg superior. Note que uma matriz de Householder é necessariamente ortogonal, então $H$ é ortogonalmente semelhante à $A$.


### Suposições:


* Se $H$ é Hessenberg superior não-singular com dimensão $n \times n$ e $H = QR$ uma fatoração $QR$, então a matriz $\hat{H} = RQ$ também é Hessenberg superior.

"""

# ╔═╡ 0b634623-51cb-44de-a81e-5496ef3bfc09
md"""
### Algoritmo:
"""

# ╔═╡ 9b026ec5-483f-4c87-bc7e-c4d590e48f5e
function iterate_H(H::Matrix, n::Int64)
	Λ = H[(end-1):end, (end-1):end]
	α = argmin(x -> abs(x - H[end]), eigen(Λ).values)
	for i ∈ 1:n
		Q, R = qr(H - α * I)
		H = R * Q + α * I		
	end
	λ = H[end]
	H = H[1:(end-1), 1:(end-1)]
	return(H = H, valor = λ)
end

# ╔═╡ 67095b20-b7b8-4123-adbb-6523d8b0bc37
function metodo_qr_shift(A::Matrix)
	H = Matrix(hessenberg(A).H)
	σ = Float64[]
	iter = 0
	while true
		n = size(H, 1)
		iter += n
		if n > 1
			H, λ = iterate_H(H, n)
			push!(σ, λ)
		else
			push!(σ, H[1])
			return(valor = Diagonal(σ), resultado = "Convergiu em $(iter) iterações")
		end
	end
end

# ╔═╡ cb52a25e-35fd-46fa-a77a-0686d48ae6ad
metodo_qr_shift(A)

# ╔═╡ 826bbc0f-ccba-472e-b41e-8ab0a7f58718
md"""
## Exemplo 1:
"""

# ╔═╡ bdcfc195-5658-49a8-b198-b4e1510db1c6
begin
	C = [4.0 0.0 0.0; 0.0 3.0 0.0; 0.0 0.0 1.0]
	latexify(C)
end

# ╔═╡ 1a95af2c-ef18-4ca8-834a-1b9f6f17cad8
eigen(C)

# ╔═╡ 8e9f2e2a-9ec1-474f-8f8d-11ea2de0a806
metodo_potencia(C)

# ╔═╡ dfc0fb39-6f4e-40e5-98a2-0c3f21b1ce34
metodo_potencia_inversa(C)

# ╔═╡ 37d306c1-fe90-44df-86d4-3312d4b1d618
metodo_potencia_shift(C, metodo_potencia(C).valor)

# ╔═╡ e390ad01-722a-4905-9644-1a6d231c32f3
metodo_potencia_shift(C, metodo_potencia_inversa(C).valor)

# ╔═╡ 6bda47c4-f368-4484-a2ef-ccf6d5576f8d
metodo_potencia_inversa_shift(C, metodo_potencia(C).valor)

# ╔═╡ 4fae8784-4272-4508-a486-bb7327114228
metodo_potencia_inversa_shift(C, metodo_potencia_inversa(C).valor)

# ╔═╡ a406b935-35cf-44b9-9d5c-c8f848790159
metodo_shift_rayleigh(C)

# ╔═╡ 26a19e11-12bd-4c6f-a330-68eaae476535
md"""
**Nota:** Às vezes o método da iteração inversa com *shift* de Rayleigh está retornando erro devido à singularidade da matriz $(A - \rho I)$ quando o $\rho$ inicial é $\epsilon$-próximo de um autovalor de $A$.
"""

# ╔═╡ 3c8e4565-2f6b-41fa-8168-9d9affdd4188
metodo_qr(C)

# ╔═╡ 335bdbfa-ce58-4752-9974-b093812a78d5
metodo_qr_shift(C)

# ╔═╡ d6e09638-3ecc-4032-bc6d-734aeb65502a
md"""
## Exemplo 2:
"""

# ╔═╡ 37f8cd78-8b58-4915-9e82-62d68254a92b
begin
	D = [-2.0 5.0 9.0 12.0; 13.0 -2.0 5.0 17.0; 34.0 5.0 0.0 -11.0; 12.0 15.0 -43.0 67.0]
	latexify(D)
end

# ╔═╡ 29987fcf-9369-4431-8883-4768fec4241a
eigen(D)

# ╔═╡ 01c6408c-6465-4306-92f9-816e378f9c9f
metodo_potencia(D)

# ╔═╡ d2e3b945-3879-4db1-9b0b-a298193ad077
metodo_potencia_inversa(D)

# ╔═╡ 2bfffa99-0cb3-430d-8d7b-207309fa0a69
metodo_potencia_shift(D, metodo_potencia(D).valor)

# ╔═╡ 2438b171-ea0b-47ec-b0c1-d10c967b1ba5
metodo_potencia_shift(D, metodo_potencia_inversa(D).valor)

# ╔═╡ ad6234dd-d431-437b-b5f8-daf82352bd83
metodo_potencia_inversa_shift(D, metodo_potencia(D).valor)

# ╔═╡ 7b95a1fe-1756-47bd-800b-4bccd58c54f7
metodo_potencia_inversa_shift(D, metodo_potencia_inversa(D).valor)

# ╔═╡ 297fff3e-899e-48f6-8915-d71017deb42b
metodo_shift_rayleigh(D)

# ╔═╡ f1bcfa54-d9d1-4e27-843a-5ac8994e55b1
metodo_qr(D)

# ╔═╡ 03e301d0-1042-4799-b759-c92be086a637
metodo_qr_shift(D)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Chain = "8be319e6-bccf-4806-a6f7-6fae938471bc"
Gadfly = "c91e804a-d5a3-530f-b6f0-dfbca275c004"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
Chain = "~0.5.0"
Gadfly = "~1.3.4"
LaTeXStrings = "~1.3.0"
Latexify = "~0.15.17"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.2"
manifest_format = "2.0"
project_hash = "1559e7754ba882b8fa9639cbfce56b6a037db28a"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "69f7020bd72f069c219b5e8c236c1fa90d2cb409"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.2.1"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "195c5505521008abea5aee4f96930717958eac6f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.4.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.CategoricalArrays]]
deps = ["DataAPI", "Future", "Missings", "Printf", "Requires", "Statistics", "Unicode"]
git-tree-sha1 = "5084cc1a28976dd1642c9f337b28a3cb03e0f7d2"
uuid = "324d7699-5711-5eae-9e2f-1d82baa6b597"
version = "0.10.7"

[[deps.Chain]]
git-tree-sha1 = "8c4920235f6c561e401dfe569beb8b924adad003"
uuid = "8be319e6-bccf-4806-a6f7-6fae938471bc"
version = "0.5.0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e7ff6cadf743c098e08fca25c91103ee4303c9bb"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.6"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "38f7a08f19d8810338d4f5085211c7dfa5d5bdd8"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.4"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "3ca828fe1b75fa84b021a7860bd039eaea84d2f2"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.3.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[deps.Compose]]
deps = ["Base64", "Colors", "DataStructures", "Dates", "IterTools", "JSON", "LinearAlgebra", "Measures", "Printf", "Random", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "d853e57661ba3a57abcdaa201f4c9917a93487a2"
uuid = "a81c6b42-2e10-5240-aca2-a61377ecd94b"
version = "0.9.4"

[[deps.Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[deps.CoupledFields]]
deps = ["LinearAlgebra", "Statistics", "StatsBase"]
git-tree-sha1 = "6c9671364c68c1158ac2524ac881536195b7e7bc"
uuid = "7ad07ef1-bdf2-5661-9d2b-286fd4296dac"
version = "0.2.0"

[[deps.DataAPI]]
git-tree-sha1 = "e08915633fcb3ea83bf9d6126292e5bc5c739922"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.13.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.Distances]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "3258d0659f812acde79e8a74b11f17ac06d0ca04"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.7"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "bee795cdeabc7601776abbd6b9aac2ca62429966"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.77"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "90630efff0894f8142308e334473eba54c433549"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.5.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "802bfc139833d2ba893dd9e62ba1767c88d708ae"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.5"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.Gadfly]]
deps = ["Base64", "CategoricalArrays", "Colors", "Compose", "Contour", "CoupledFields", "DataAPI", "DataStructures", "Dates", "Distributions", "DocStringExtensions", "Hexagons", "IndirectArrays", "IterTools", "JSON", "Juno", "KernelDensity", "LinearAlgebra", "Loess", "Measures", "Printf", "REPL", "Random", "Requires", "Showoff", "Statistics"]
git-tree-sha1 = "13b402ae74c0558a83c02daa2f3314ddb2d515d3"
uuid = "c91e804a-d5a3-530f-b6f0-dfbca275c004"
version = "1.3.4"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.Hexagons]]
deps = ["Test"]
git-tree-sha1 = "de4a6f9e7c4710ced6838ca906f81905f7385fd6"
uuid = "a1b4810d-1bce-5fbd-ac56-80944d57a21f"
version = "0.2.0"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions", "Test"]
git-tree-sha1 = "709d864e3ed6e3545230601f94e11ebc65994641"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.11"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "842dd89a6cb75e02e85fdd75c760cdc43f5d6863"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.14.6"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "49510dfcb407e572524ba94aeae2fced1f3feb0f"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.8"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.Juno]]
deps = ["Base64", "Logging", "Media", "Profile"]
git-tree-sha1 = "07cb43290a840908a771552911a6274bc6c072c7"
uuid = "e5e0dc1b-0480-54bc-9374-aad01c23163d"
version = "0.8.4"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "9816b296736292a80b9a3200eb7fbb57aaa3917a"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.5"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "ab9aa169d2160129beb241cb2750ca499b4e90e9"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.17"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Loess]]
deps = ["Distances", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "46efcea75c890e5d820e670516dc156689851722"
uuid = "4345ca2d-374a-55d4-8d30-97f9976e7612"
version = "0.5.4"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "94d9c52ca447e23eac0c0f074effbcd38830deb5"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.18"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "2ce8695e1e699b68702c03402672a69f54b8aca9"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2022.2.0+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[deps.Media]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "75a54abd10709c01f1b86b84ec225d26e840ed58"
uuid = "e89f7d12-3494-54d1-8411-f7d8b9ae1f27"
version = "0.5.0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "a7c3d1da1189a1c2fe843a3bfa04d18d20eb3211"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "f71d8950b724e9ff6110fc948dff5a329f901d64"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.8"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "cf494dca75a69712a72b80bc48f59dcf3dea63ec"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.16"

[[deps.Parsers]]
deps = ["Dates", "SnoopPrecompile"]
git-tree-sha1 = "cceb0257b662528ecdf0b4b4302eb00e767b38e7"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "97aa253e65b784fd13e83774cadc95b38011d734"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.6.0"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "dc84268fe0e3335a62e315a3a7cf2afa7178a734"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.3"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SnoopPrecompile]]
git-tree-sha1 = "f604441450a3c0569830946e5b33b78c928e1a85"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.1"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "d75bda01f8c31ebb72df80a46c88b25d1c79c56d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.7"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "f86b3a049e5d05227b10e15dbb315c5b90f14988"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.9"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6b7ba252635a5eff6a0b0664a41ee140a1c9e72a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f9af7f195fb13589dd2e2d57fdb401717d2eb1f6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.5.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "5783b877201a82fc0014cbf381e7e6eb130473a4"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.0.1"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╟─88948702-62b7-11ed-0862-a94108822a9a
# ╟─f17f1b3c-0cb5-476e-a756-fc634ed53ec0
# ╟─cd5be5c8-ec66-49cf-a35f-581f6945e2de
# ╟─9fcefb8c-dd1b-4d4f-805b-4860ceda0499
# ╟─3274165a-6d9e-4e88-a22e-b5271ba90c09
# ╟─73be5700-36ec-49e2-af59-d3e6ec205fbd
# ╟─7d9eaf4e-bd8f-4a32-8faf-538aa649fb0e
# ╠═bd9c73d9-7e66-4666-9944-3d0f02f841d2
# ╠═4e964318-5a08-4e4e-9660-b75c3048b008
# ╟─11d2b341-ee95-4d34-939f-8f7db8d2e8d0
# ╟─fa1eea07-e0e8-48c6-9c9d-04f6ff5fff8f
# ╠═b142b019-9deb-425a-b906-78d2108cd6e4
# ╠═59e3c1ba-34d7-4fc3-ac8c-317a9c55b1c1
# ╟─eb7bbeae-0401-4702-b8ce-e24c1ea28e8f
# ╟─6b464cd7-a759-4507-8f82-91737c8ee7f5
# ╟─898d5516-6ab7-46d5-9d7e-628096e8b8cf
# ╠═93225e28-7277-4906-a49a-16c6f8e6cdaa
# ╟─ead9093d-07fb-4c43-8985-5a114b97071c
# ╠═e4e6d0c7-4ed8-4a33-a064-7632e0893b34
# ╟─67eec47c-cf99-430e-a618-0df548200a82
# ╠═c2afceac-1fe7-40bc-9e29-2127edc6947f
# ╠═a6de036e-7d32-47ca-b0ab-a563e976a6d5
# ╟─947e5d14-ce3e-4141-bb96-e3d21d15f35d
# ╟─55b3af7e-cf75-4ef7-a5aa-f038ef61b0ff
# ╠═d323edc9-0f07-4b26-80af-37ffd417deaf
# ╠═3bf0092d-d8b1-4610-af55-c93bb42b307b
# ╠═e21a239d-176d-4da7-a71e-25e9f4388605
# ╠═f0fe0a76-6aa2-45dd-b4d0-70dfd1618516
# ╠═04d987cc-bbbd-4a1b-a48b-25a9ce107af9
# ╠═d02eb301-d769-438d-b758-35e2a75499dc
# ╠═f054bc30-ecae-444e-ad28-573991170f33
# ╟─57f5909b-3e54-42dd-a2b8-d6f2f8591f23
# ╟─bb6661ad-b93e-4cbc-a903-ec7149a74b9b
# ╟─b0974dd6-3617-45c3-a258-7c6113461f90
# ╠═c8709478-f17f-4674-bd24-15a22672a7df
# ╠═5b5709c2-ba16-4569-84ca-d04524102954
# ╠═e4ce0e06-6409-4e6e-bc21-e68488f45d0f
# ╠═8a142039-b13d-42e1-a68e-ec415d37d16a
# ╟─ad2a2fd8-d719-496e-be37-e1a7e85992df
# ╟─8b5f7ffa-a04a-4fbd-a560-51e45b975c35
# ╟─47db7ece-5379-4323-8357-66353af0e747
# ╠═ba5264a5-565a-4ae7-afa7-7800c6d556ea
# ╠═38869197-dd65-4c35-a89f-0a535a08e63c
# ╟─4f85f994-4fc9-4e63-a56a-ed5da19328cb
# ╟─0b634623-51cb-44de-a81e-5496ef3bfc09
# ╠═9b026ec5-483f-4c87-bc7e-c4d590e48f5e
# ╠═67095b20-b7b8-4123-adbb-6523d8b0bc37
# ╠═cb52a25e-35fd-46fa-a77a-0686d48ae6ad
# ╟─826bbc0f-ccba-472e-b41e-8ab0a7f58718
# ╟─bdcfc195-5658-49a8-b198-b4e1510db1c6
# ╠═1a95af2c-ef18-4ca8-834a-1b9f6f17cad8
# ╠═8e9f2e2a-9ec1-474f-8f8d-11ea2de0a806
# ╠═dfc0fb39-6f4e-40e5-98a2-0c3f21b1ce34
# ╠═37d306c1-fe90-44df-86d4-3312d4b1d618
# ╠═e390ad01-722a-4905-9644-1a6d231c32f3
# ╠═6bda47c4-f368-4484-a2ef-ccf6d5576f8d
# ╠═4fae8784-4272-4508-a486-bb7327114228
# ╠═a406b935-35cf-44b9-9d5c-c8f848790159
# ╟─26a19e11-12bd-4c6f-a330-68eaae476535
# ╠═3c8e4565-2f6b-41fa-8168-9d9affdd4188
# ╠═335bdbfa-ce58-4752-9974-b093812a78d5
# ╟─d6e09638-3ecc-4032-bc6d-734aeb65502a
# ╟─37f8cd78-8b58-4915-9e82-62d68254a92b
# ╠═29987fcf-9369-4431-8883-4768fec4241a
# ╠═01c6408c-6465-4306-92f9-816e378f9c9f
# ╠═d2e3b945-3879-4db1-9b0b-a298193ad077
# ╠═2bfffa99-0cb3-430d-8d7b-207309fa0a69
# ╠═2438b171-ea0b-47ec-b0c1-d10c967b1ba5
# ╠═ad6234dd-d431-437b-b5f8-daf82352bd83
# ╠═7b95a1fe-1756-47bd-800b-4bccd58c54f7
# ╠═297fff3e-899e-48f6-8915-d71017deb42b
# ╠═f1bcfa54-d9d1-4e27-843a-5ac8994e55b1
# ╠═03e301d0-1042-4799-b759-c92be086a637
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
