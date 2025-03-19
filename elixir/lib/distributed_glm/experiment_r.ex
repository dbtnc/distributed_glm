defmodule DistributedGLM.ExperimentR do
  alias DistributedGLM.Simulation
  alias NimbleCSV.RFC4180, as: CSV

  @models [:lm, :glm]

  # If the CSV file is large for the system, a streaming
  # approach should be used instead
  def run(n \\ 2) do
    Enum.each(@models, fn m -> model_run(m, n) end)
  end

  defp model_run(m, n) when m in @models do
    data = model_data(m)
    beta = %{coefficients: Nx.tensor(model_beta(m), type: :f64)}
    experiment_result = experiment(m).(n, data, beta)
    print_result(m, experiment_result)
    cleanup(m)
  end

  defp cleanup(m) do
    File.rm!("../#{m}_beta.csv")
    File.rm!("../#{m}_mm.csv")
  end

  defp print_result(m, experiment_result) when m in @models do
    Atom.to_string(m)
    |> String.upcase()
    |> then(&IO.puts("#{&1} experiment similar: #{elem(experiment_result, 0)}"))
  end

  defp experiment(:lm), do: &Simulation.linear_model/3
  defp experiment(:glm), do: &Simulation.generalized_linear_model/3

  def model_beta(m) when m in @models do
    Path.expand("../#{m}_beta.csv")
    |> File.read!()
    |> CSV.parse_string()
    |> Enum.map(fn [num] ->
      [Float.parse(num) |> elem(0)]
    end)
  end

  def model_data(m) when m in @models do
    Path.expand("../#{m}_mm.csv")
    |> File.stream!()
    |> CSV.parse_stream()
    # |> Stream.transform(%{x: [], y: []}, fn row, acc ->
    #  {y, x} = Enum.map(row, fn num -> 
    #    Float.parse(num) |> elem(0)
    #  end)
    #  |> List.pop_at(-1)
    #  {%{x: [x|acc.x], y: [y|acc.y]} , %{x: [x|acc.x], y: [y|acc.y]}}
    # end)
    |> Stream.map(fn row ->
      Enum.map(row, fn num ->
        Float.parse(num) |> elem(0)
      end)
    end)
    |> Enum.reduce(%{x: [], y: []}, fn row, acc ->
      {y, x} = List.pop_at(row, -1)
      %{x: [x | acc.x], y: [[y] | acc.y]}
    end)
  end
end
