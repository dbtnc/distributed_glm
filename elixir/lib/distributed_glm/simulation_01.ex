defmodule DistributedGLM.Simulation01 do
  alias DistributedGLM.Simulation
  alias NimbleCSV.RFC4180, as: CSV

  @models [:lm, :glm]

  # If the CSV file is large for the system, a streaming
  # approach should be used instead
  def run(n \\ 2) do
    Path.wildcard("../{#{Enum.join(@models, ",")}}_*_*_*_mm.csv")
    |> Enum.each(fn sim ->
      captures =
        Regex.named_captures(~r/..\/(?<m>[glm|lm]*)_(?<row>\d*)_(?<col>\d*)_(?<rep>\d*)/, sim)

      setting = "#{captures["m"]}_#{captures["row"]}_#{captures["col"]}_#{captures["rep"]}"
      model_run(setting, String.to_existing_atom(captures["m"]), n)
    end)
  end

  defp model_run(setting, m, n) when m in @models do
    data = model_data(setting)
    beta = %{coefficients: Nx.tensor(model_beta(setting), type: :f64)}
    {experiment_result, _coefficients} = experiment(m).(n, data, beta)
    print_result(setting, experiment_result)
  end

  defp print_result(m, experiment_result) when m in @models do
    Atom.to_string(m)
    |> String.upcase()
    |> then(&IO.puts("#{&1} experiment similar: #{experiment_result}"))
  end

  defp print_result(m, experiment_result) when is_binary(m) do
    m
    |> String.upcase()
    |> then(&IO.puts("#{&1} experiment similar: #{experiment_result}"))
  end

  defp experiment(:lm), do: &Simulation.linear_model/3
  defp experiment(:glm), do: &Simulation.generalized_linear_model/3

  def model_beta(setting) do
    Path.expand("../#{setting}_beta.csv")
    |> File.read!()
    |> CSV.parse_string()
    |> Enum.map(fn [num] ->
      [Float.parse(num) |> elem(0)]
    end)
  end

  def model_data(setting) do
    Path.expand("../#{setting}_mm.csv")
    |> File.stream!()
    |> CSV.parse_stream()
    |> Stream.map(fn row ->
      Enum.map(row, fn num ->
        Float.parse(num) |> elem(0)
      end)
    end)
    |> Enum.reduce(%{x: [], y: []}, fn row, acc ->
      {y, x} = List.pop_at(row, 0)
      %{x: [x | acc.x], y: [[y] | acc.y]}
    end)
    |> add_intercept()
  end

  defp add_intercept(data = %{x: x, y: _}) do
    Enum.reduce_while(x, false, fn e, _acc ->
      # Should compare with a nearly equal function, but ok for this
      if List.first(e) == 1.0 do
        {:cont, false}
      else
        {:halt, true}
      end
    end)
    |> then(&add_intercept(data, &1))
  end

  defp add_intercept(data = %{x: _, y: _}, false), do: data

  defp add_intercept(data = %{x: x, y: _}, true) do
    Enum.map(x, fn e ->
      [1.0 | e]
    end)
    |> then(&Map.put(data, :x, &1))
  end
end
