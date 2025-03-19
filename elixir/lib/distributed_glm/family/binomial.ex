defmodule DistributedGLM.Family.Binomial do
  import Nx.Defn

  defn linkinv(eta) do
    # exp_eta = Nx.map(eta, fn e ->
    #  cond do
    #    e < Nx.f64(-30) -> Nx.Constants.epsilon({:f, 64})
    #    e > Nx.f64(30) -> 1 / Nx.Constants.epsilon({:f, 64})
    #    true -> Nx.exp(e)
    #  end
    # end)
    # exp_eta / (1 + exp_eta)

    exp = Nx.exp(eta)
    exp / (1 + exp)
  end

  defn mu_eta(eta) do
    op = 1 + Nx.exp(eta)
    Nx.exp(eta) / (op * op)

    # Nx.map(eta, fn e ->
    #  if e < Nx.f64(-30) or e > Nx.f64(30) do
    #    Nx.Constants.epsilon({:f, 64})
    #  else
    #    op = 1 + Nx.exp(e)
    #    Nx.exp(e) / (op * op)
    #  end
    # end)
  end

  defn variance(mu) do
    mu * (1 - mu)
  end
end
