defmodule DistributedGLM.GeneralizedLinearModelTest do
  use ExUnit.Case
  doctest DistributedGLM.GeneralizedLinearModel

  alias DistributedGLM.Family.Binomial
  alias DistributedGLM.Family.Gaussian

  alias DistributedGLM.GeneralizedLinearModel
  alias DistributedGLM.LinearModel

  test "with family=Gaussian is equal to Linear Model" do
    y_1 = Nx.tensor([[57], [55], [2]]) |> Nx.multiply(Nx.f64(1.0))
    x_1 = Nx.tensor([[15], [4], [13]]) |> Nx.multiply(Nx.f64(1.0))
    x_1 = Nx.concatenate([Nx.broadcast(1, Nx.shape(x_1)), x_1], axis: 1)

    y_2 = Nx.tensor([[5], [50], [25]]) |> Nx.multiply(Nx.f64(1.0))
    x_2 = Nx.tensor([[3], [9], [1]]) |> Nx.multiply(Nx.f64(1.0))
    x_2 = Nx.concatenate([Nx.broadcast(1, Nx.shape(x_2)), x_2], axis: 1)

    x = Nx.concatenate([x_1, x_2])
    y = Nx.concatenate([y_1, y_2])

    lm = LinearModel.fit(x, y)
    glm = GeneralizedLinearModel.fit(x, y, family: Gaussian)

    assert Nx.all_close(lm.coefficients, glm.coefficients) |> Nx.to_number() == 1
  end

  test "Binomial family" do
    {x, y} = gen_data(:binomial)
    glm = GeneralizedLinearModel.fit(x, y, family: Binomial)
    # from R glm()
    beta_hat = Nx.tensor([[-18.796313], [1.843344]])
    assert Nx.all_close(glm.coefficients, beta_hat) |> Nx.to_number() == 1
  end

  defp gen_data(:binomial) do
    y =
      Nx.tensor(
        [
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0
        ],
        type: :f64
      )
      |> Nx.reshape({80, 1})

    x =
      Nx.tensor(
        [
          13.192708,
          14.448090,
          9.649417,
          10.320116,
          11.647111,
          13.650080,
          14.084886,
          12.802728,
          12.112105,
          12.037028,
          13.965425,
          15.977136,
          12.453174,
          15.544261,
          11.953416,
          13.113029,
          9.641135,
          12.180368,
          12.420320,
          11.397851,
          12.907001,
          11.365693,
          10.844742,
          14.534238,
          15.422245,
          13.413604,
          12.944656,
          9.860943,
          11.712985,
          16.808641,
          14.413471,
          13.190056,
          10.360630,
          10.401697,
          13.787254,
          10.377768,
          13.894956,
          10.628888,
          14.807852,
          13.198363,
          5.837473,
          10.179199,
          6.625756,
          3.322105,
          9.080811,
          3.816459,
          6.389127,
          4.613805,
          7.655697,
          5.843509,
          7.382162,
          8.495226,
          8.414558,
          9.663929,
          9.625316,
          5.727101,
          5.629926,
          7.671450,
          7.717170,
          9.424363,
          6.922032,
          8.777796,
          9.283351,
          8.233282,
          6.498535,
          4.674278,
          9.026015,
          10.186052,
          8.900973,
          3.947789,
          11.527118,
          7.360485,
          7.442523,
          7.757168,
          9.294623,
          6.753300,
          10.380745,
          5.958523,
          6.278818,
          10.985235
        ],
        type: :f64
      )
      |> Nx.reshape({80, 1})

    x = Nx.concatenate([Nx.broadcast(1, Nx.shape(x)), x], axis: 1)

    {x, y}
  end
end
