defmodule DistributedGLM.MixProject do
  use Mix.Project

  def project do
    [
      app: :distributed_glm,
      version: "0.1.0",
      elixir: "~> 1.16",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      mod: {DistributedGLM.Application, []},
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:nx, "~> 0.6.0"},
      {:exla, "~> 0.6.0"},
      {:nimble_csv, "~> 1.1"}
    ]
  end
end
