defmodule DistributedGLM.GeneralizedLinearModel do
  import Nx.Defn

  alias DistributedGLM.Family.Binomial
  alias DistributedGLM.Family.Gaussian

  def default_maxit, do: 25
  def default_tol, do: 1.0e-10

  @derive {Nx.Container, containers: [:r_local, :coefficients]}
  defstruct [:r_local, :coefficients, :family, :iter]
  # TODO: Family should be a strcut instead of a Module to allow for different
  #       link functions

  deftransform fit(x, y, opts \\ []) when is_list(opts) do
    # TODO: Refactor to something like NimbleOptions
    opts =
      opts
      |> Keyword.put_new(:family, Gaussian)
      |> Keyword.put_new(:maxit, 25)
      |> Keyword.put_new(:tol, 1.0e-10)

    {r_local, beta, iter} =
      case opts[:family] do
        Gaussian ->
          fit_gaussian_n(x, y, opts)

        Binomial ->
          fit_binomial_n(x, y, opts)

        _ ->
          fit_gaussian_n(x, y, opts)
      end

    %__MODULE__{r_local: r_local, coefficients: beta, family: opts[:family], iter: iter}
  end

  # TODO: Refactor (fit|update)_family_n to single (fit|update)_n

  defn distributed_gaussian_single_iter_n(x, y, beta) do
    eta = Nx.dot(x, beta)
    mu = Gaussian.linkinv(eta)
    dmu = Gaussian.mu_eta(eta)
    z = eta + (y - mu) / dmu
    w = dmu * dmu / Gaussian.variance(mu)

    x_tilde = Nx.sqrt(w) * x
    z_tilde = Nx.sqrt(w) * z

    {_, r_local} =
      [x_tilde, z_tilde]
      |> Nx.concatenate(axis: 1)
      |> Nx.LinAlg.qr()

    r_local
  end

  defn distributed_binomial_single_iter_n(x, y, beta) do
    eta = Nx.dot(x, beta)
    mu = Binomial.linkinv(eta)
    dmu = Binomial.mu_eta(eta)
    z = eta + (y - mu) / dmu
    w = dmu * dmu / Binomial.variance(mu)

    x_tilde = Nx.sqrt(w) * x
    z_tilde = Nx.sqrt(w) * z

    {_, r_local} =
      [x_tilde, z_tilde]
      |> Nx.concatenate(axis: 1)
      |> Nx.LinAlg.qr()

    r_local
  end

  defn distributed_gaussian_single_solve_n(
         r_local_with_all_r_remotes,
         beta,
         total_nrow,
         maxit,
         tol,
         iter
       ) do
    beta_old = beta
    {r_local, beta} = ols_n(r_local_with_all_r_remotes)

    vcov = vcov(r_local, Gaussian, total_nrow)
    delta = (beta_old - beta) / Nx.sqrt(Nx.take_diagonal(vcov))
    diff = delta |> Nx.abs() |> Nx.reduce_max()
    stop = stop(maxit, tol, iter, diff)

    {r_local, beta, stop}
  end

  defn distributed_binomial_single_solve_n(
         r_local_with_all_r_remotes,
         beta,
         total_nrow,
         maxit,
         tol,
         iter
       ) do
    beta_old = beta
    {r_local, beta} = ols_n(r_local_with_all_r_remotes)

    vcov = vcov(r_local, Binomial, total_nrow)
    delta = (beta_old - beta) / Nx.sqrt(Nx.take_diagonal(vcov))
    diff = delta |> Nx.abs() |> Nx.reduce_max()
    stop = stop(maxit, tol, iter, diff)

    {r_local, beta, stop}
  end

  defnp fit_gaussian_n(x, y, opts) do
    _family = opts[:family]
    maxit = opts[:maxit]
    tol = opts[:tol]

    type = Nx.type(x)
    {_, c} = Nx.shape(x)
    beta = Nx.broadcast(0, {c, 1}) * Nx.tensor(1, type: type)
    r_local = Nx.broadcast(0, {c + 1, c + 1}) * Nx.tensor(1, type: type)
    stop = Nx.u8(0)
    iter = Nx.s64(1)

    while {_r_local = r_local, beta, x, y, maxit, tol, stop, iter}, Nx.less(stop, 1) do
      # loop
      eta = Nx.dot(x, beta)
      mu = Gaussian.linkinv(eta)
      dmu = Gaussian.mu_eta(eta)
      z = eta + (y - mu) / dmu
      w = dmu * dmu / Gaussian.variance(mu)

      x_tilde = Nx.sqrt(w) * x
      z_tilde = Nx.sqrt(w) * z

      beta_old = beta

      {r_local, beta} =
        [x_tilde, z_tilde]
        |> Nx.concatenate(axis: 1)
        |> ols_n()

      vcov = vcov(r_local, Gaussian, Nx.shape(x) |> elem(0))
      delta = (beta_old - beta) / Nx.sqrt(Nx.take_diagonal(vcov))
      diff = delta |> Nx.abs() |> Nx.reduce_max()
      stop = stop(maxit, tol, iter, diff)

      iter =
        if stop do
          iter
        else
          iter + 1
        end

      {r_local, beta, x, y, maxit, tol, stop, iter}
    end
    |> then(&{elem(&1, 0), elem(&1, 1), elem(&1, 7)})
  end

  defnp fit_binomial_n(x, y, opts) do
    _family = opts[:family]
    maxit = opts[:maxit] - 1
    tol = opts[:tol] * Nx.f64(1)

    type = Nx.type(x)
    {_, c} = Nx.shape(x)
    beta = Nx.broadcast(0, {c, 1}) * Nx.tensor(1, type: type)
    r_local = Nx.broadcast(0, {c + 1, c + 1}) * Nx.tensor(1, type: type)
    stop = Nx.u8(0)
    iter = Nx.s64(1)

    while {_r_local = r_local, beta, x, y, maxit, tol, stop, iter}, Nx.less(stop, 1) do
      # loop
      eta = Nx.dot(x, beta)
      mu = Binomial.linkinv(eta)
      dmu = Binomial.mu_eta(eta)
      z = eta + (y - mu) / dmu
      w = dmu * dmu / Binomial.variance(mu)

      x_tilde = Nx.sqrt(w) * x
      z_tilde = Nx.sqrt(w) * z

      beta_old = beta

      {r_local, beta} =
        [x_tilde, z_tilde]
        |> Nx.concatenate(axis: 1)
        |> ols_n()

      vcov = vcov(r_local, Binomial, Nx.shape(x) |> elem(0))
      delta = (beta_old - beta) / Nx.sqrt(Nx.take_diagonal(vcov))
      diff = delta |> Nx.abs() |> Nx.reduce_max()
      stop = stop(maxit, tol, iter, diff)

      iter =
        if stop do
          iter
        else
          iter + 1
        end

      {r_local, beta, x, y, maxit, tol, stop, iter}
    end
    |> then(&{elem(&1, 0), elem(&1, 1), elem(&1, 7)})
  end

  defnp stop(maxit, tol, iter, diff) do
    maxit <= iter or diff < tol
  end

  defnp vcov(r_local, family, total_nrow) do
    r = r_local[[0..-2//1, 0..-2//1]]
    rss = r_local[[-1, -1]]

    ncol = Nx.shape(r_local) |> elem(1)
    inv_r = Nx.transpose(r) |> Nx.dot(r) |> Nx.LinAlg.invert()

    dispersion =
      case family do
        Binomial -> 1
        _ -> rss * rss / (total_nrow - ncol)
      end

    inv_r * dispersion
  end

  defnp ols_n(r_xy_or_xy) do
    {_q_s, r_s} = Nx.LinAlg.qr(r_xy_or_xy)

    r = r_s[[0..-2//1, 0..-2//1]]
    theta = r_s[[0..-2//1, -1..-1//1]]
    b = Nx.LinAlg.triangular_solve(r, theta, lower: false)

    {r_s, b}
  end
end
