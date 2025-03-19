defmodule DistributedGLM.LinearModel do
  import Nx.Defn

  @derive {Nx.Container, containers: [:r_local, :coefficients]}
  defstruct [:r_local, :coefficients]

  deftransform fit(x, y, _opts \\ []) do
    {r_local, beta} = fit_n(x, y)
    %__MODULE__{r_local: r_local, coefficients: beta}
  end

  defn deviance(lm) do
    lm.r_local[[-1..-1//1, -1..-1//1]]
    |> Nx.flatten()
  end

  defnp fit_n(x, y) do
    [x, y]
    |> Nx.concatenate(axis: 1)
    |> ols_n()
  end

  deftransform update(lm, x, y, _opts \\ []) do
    {r_local, beta} = update_n(lm.r_local, x, y)
    %__MODULE__{r_local: r_local, coefficients: beta}
  end

  deftransform update_distributed(lm, r_remote, _opts \\ []) do
    {r_local, beta} = update_distributed_n(lm.r_local, r_remote)
    %__MODULE__{r_local: r_local, coefficients: beta}
  end

  defn update_distributed(r_local_with_all_r_remotes) do
    ols_n(r_local_with_all_r_remotes)
  end

  defnp update_distributed_n(r_local, r_remote) do
    [r_local, r_remote]
    |> Nx.concatenate()
    |> ols_n()
  end

  defnp update_n(r_local, x, y) do
    [x, y]
    |> Nx.concatenate(axis: 1)
    |> then(&Nx.concatenate([r_local, &1]))
    |> ols_n()
  end

  defnp ols_n(r_xy_or_xy) do
    {_q_s, r_s} = Nx.LinAlg.qr(r_xy_or_xy)

    r = r_s[[0..-2//1, 0..-2//1]]
    theta = r_s[[0..-2//1, -1..-1//1]]
    b = Nx.LinAlg.triangular_solve(r, theta, lower: false)

    {r_s, b}
  end
end
