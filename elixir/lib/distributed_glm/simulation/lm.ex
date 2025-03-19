defmodule DistributedGLM.Simulation.LM do
  use GenServer

  alias DistributedGLM.LinearModel

  def start_link(default) when is_list(default) do
    GenServer.start_link(__MODULE__, default)
  end

  def lm(pid) do
    GenServer.call(pid, :lm)
  end

  def start_simulation(pid, nodes) do
    GenServer.cast(pid, {:start_simulation, nodes})
  end

  def concat_r(pid, from, r_remote) do
    GenServer.cast(pid, {:concat_r, from, r_remote})
  end

  @impl true
  def init([x, y]) do
    lm = LinearModel.fit(x, y)

    initial_state = %{
      model: lm,
      r_remotes: %{},
      nodes: []
    }

    {:ok, initial_state}
  end

  @impl true
  def handle_cast({:start_simulation, nodes}, state) do
    nodes =
      Enum.reduce(nodes, [], fn node, acc ->
        if node != self() do
          concat_r(node, self(), state.model.r_local)
          [node | acc]
        else
          acc
        end
      end)

    state = %{state | nodes: nodes}

    {:noreply, state}
  end

  @impl true
  def handle_cast({:concat_r, from, r_remote}, state) do
    if Map.has_key?(state.r_remotes, from) do
      {:noreply, state}
    else
      r_remotes = Map.put_new(state.r_remotes, from, r_remote)

      state =
        if length(state.nodes) == length(Map.keys(r_remotes)) do
          model =
            r_remotes
            |> Map.values()
            |> Nx.concatenate()
            |> then(&LinearModel.update_distributed(state.model, &1))

          {:ok, parent} = Registry.meta(Registry.Simulation, :parent)
          send(parent, :simulation_is_done)

          %{state | r_remotes: r_remotes, model: model}
        else
          %{state | r_remotes: r_remotes}
        end

      {:noreply, state}
    end
  end

  @impl true
  def handle_call(:lm, _from, state) do
    {:reply, state.model, state}
  end
end
