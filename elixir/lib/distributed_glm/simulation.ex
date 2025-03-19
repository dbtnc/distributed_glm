defmodule DistributedGLM.Simulation do
  alias DistributedGLM.LinearModel
  alias DistributedGLM.GeneralizedLinearModel
  alias DistributedGLM.Simulation.LM
  alias DistributedGLM.Simulation.GLM

  # data should be a map of %{x => [[]], y => []}
  def linear_model(n, data) when is_map(data) do
    nodes = start_run(n, data, LM)
    central_lm = LinearModel.fit(Nx.tensor(data.x, type: :f64), Nx.tensor(data.y, type: :f64))
    check?(central_lm, nodes, &LM.lm/1)
  end

  def linear_model(n, data, central_lm) when is_map(data) do
    nodes = start_run(n, data, LM)
    check?(central_lm, nodes, &LM.lm/1)
  end

  def generalized_linear_model(n, data) when is_map(data) do
    nodes = start_run(n, data, GLM)

    central_glm =
      GeneralizedLinearModel.fit(
        Nx.tensor(data.x, type: :f64),
        Nx.tensor(data.y, type: :f64),
        family: DistributedGLM.Family.Binomial
      )

    check?(central_glm, nodes, &GLM.glm/1)
  end

  def generalized_linear_model(n, data, central_glm) when is_map(data) do
    nodes = start_run(n, data, GLM)
    check?(central_glm, nodes, &GLM.glm/1)
  end

  def run(n \\ 2) do
    x = [
      [1.0, 15.0],
      [1.0, 4.0],
      [1.0, 13.0],
      [1.0, 3.0],
      [1.0, 9.0],
      [1.0, 1.0],
      [1.0, 15.0],
      [1.0, 4.0],
      [1.0, 13.0],
      [1.0, 3.0],
      [1.0, 9.0],
      [1.0, 1.0]
    ]

    y = [
      [57.0],
      [55.0],
      [2.0],
      [5.0],
      [50.0],
      [25.0],
      [57.0],
      [55.0],
      [2.0],
      [5.0],
      [50.0],
      [25.0]
    ]

    res_lm = linear_model(n, %{x: x, y: y})

    {x, y} = gen_data(:binomial)
    res_glm = generalized_linear_model(n, %{x: Nx.to_list(x), y: Nx.to_list(y)})

    [res_lm, res_glm]
  end

  defp check?(central, nodes, func) do
    res =
      Enum.map(nodes, fn node ->
        func.(node).coefficients
        |> Nx.all_close(central.coefficients)
        |> Nx.to_number()
      end)
      |> Enum.all?(fn r -> r == 1 end)

    {res, central.coefficients}
  end

  defp start_run(n, data, module) do
    y_len = length(data.y)
    ncols = List.first(data.x) |> length()

    if length(data.x) != y_len do
      raise "length(x) != length(y)"
    end

    if n * (ncols + 1) >= y_len do
      raise "split > ncols"
    end

    data_x = Nx.tensor(data.x, type: :f64)
    data_y = Nx.tensor(data.y, type: :f64)

    nodes =
      Enum.zip_reduce(
        chunk_nx(data_x, n),
        chunk_nx(data_y, n),
        [],
        fn x, y, acc ->
          {:ok, node} =
            DynamicSupervisor.start_child(
              DistributedGLM.DynamicSupervisor,
              {module, [x, y]}
            )

          [node | acc]
        end
      )

    parent = self()
    :ok = Registry.put_meta(Registry.Simulation, :parent, parent)

    Task.start_link(fn ->
      for node <- nodes do
        module.start_simulation(node, nodes)
      end
    end)

    for _node <- nodes do
      receive do
        :simulation_is_done -> :ok
      after
        # Optional timeout
        30_000 -> raise "simulation timeout"
      end
    end

    nodes
  end

  def chunk_nx(mat, 1), do: mat

  def chunk_nx(mat, n) do
    {l, _} = Nx.shape(mat)
    nsplits = div(l, n)

    {lres, mres} =
      Enum.reduce(1..(n - 1), {[], mat}, fn _s, {chunks, mat_s} ->
        {mat_chunk, mat_rest} = Nx.split(mat_s, nsplits)
        {[mat_chunk | chunks], mat_rest}
      end)

    Enum.reverse([mres | lres])
  end

  def chunk_data(list, n) do
    len = length(list)
    s = Integer.floor_div(len, n)

    if Integer.mod(len, n) == 0 do
      Enum.chunk_every(list, s)
    else
      {last, l} =
        Enum.chunk_every(list, s)
        |> List.pop_at(-1)

      List.replace_at(l, -1, List.last(l) ++ last)
    end
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
