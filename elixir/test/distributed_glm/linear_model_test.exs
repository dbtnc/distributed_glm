defmodule DistributedGLM.LinearModelTest do
  use ExUnit.Case
  doctest DistributedGLM.LinearModel

  alias DistributedGLM.LinearModel

  test "update equals fit with all values" do
    y_1 = Nx.tensor([[57], [55], [2]]) |> Nx.multiply(Nx.f64(1.0))
    x_1 = Nx.tensor([[15], [4], [13]]) |> Nx.multiply(Nx.f64(1.0))
    x_1 = Nx.concatenate([Nx.broadcast(1, Nx.shape(x_1)), x_1], axis: 1)

    y_2 = Nx.tensor([[5], [50], [25]]) |> Nx.multiply(Nx.f64(1.0))
    x_2 = Nx.tensor([[3], [9], [1]]) |> Nx.multiply(Nx.f64(1.0))
    x_2 = Nx.concatenate([Nx.broadcast(1, Nx.shape(x_2)), x_2], axis: 1)

    lm1 = LinearModel.fit(Nx.concatenate([x_1, x_2]), Nx.concatenate([y_1, y_2]))

    lm2 =
      LinearModel.fit(x_1, y_1)
      |> LinearModel.update(x_2, y_2)

    assert Nx.all_close(lm1.coefficients, lm2.coefficients) |> Nx.to_number() == 1
  end

  test "update_distributed equals fit with all values" do
    y_1 = Nx.tensor([[57], [55], [2]]) |> Nx.multiply(Nx.f64(1.0))
    x_1 = Nx.tensor([[15], [4], [13]]) |> Nx.multiply(Nx.f64(1.0))
    x_1 = Nx.concatenate([Nx.broadcast(1, Nx.shape(x_1)), x_1], axis: 1)

    y_2 = Nx.tensor([[5], [50], [25]]) |> Nx.multiply(Nx.f64(1.0))
    x_2 = Nx.tensor([[3], [9], [1]]) |> Nx.multiply(Nx.f64(1.0))
    x_2 = Nx.concatenate([Nx.broadcast(1, Nx.shape(x_2)), x_2], axis: 1)

    lm1 = LinearModel.fit(Nx.concatenate([x_1, x_2]), Nx.concatenate([y_1, y_2]))
    lm2 = LinearModel.fit(x_1, y_1)

    lm3 =
      LinearModel.fit(x_2, y_2)
      |> LinearModel.update_distributed(lm2.r_local)

    assert Nx.all_close(lm1.coefficients, lm3.coefficients) |> Nx.to_number() == 1
  end
end
