defmodule DistributedGLM.Family.Gaussian do
  import Nx.Defn

  defn linkinv(eta) do
    eta
  end

  defn mu_eta(eta) do
    Nx.broadcast(1, Nx.shape(eta))
  end

  defn variance(mu) do
    Nx.broadcast(1, Nx.shape(mu))
  end
end
