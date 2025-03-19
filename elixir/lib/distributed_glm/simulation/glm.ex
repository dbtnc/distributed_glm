defmodule DistributedGLM.Simulation.GLM do
  use GenServer

  alias DistributedGLM.Family
  alias DistributedGLM.GeneralizedLinearModel

  def start_link(default) when is_list(default) do
    GenServer.start_link(__MODULE__, default)
  end

  def glm(pid) do
    GenServer.call(pid, :glm)
  end

  def start_simulation(pid, nodes) do
    GenServer.cast(pid, {:start_simulation, nodes})
  end

  def sum_rows(pid, from, nrows) do
    GenServer.cast(pid, {:sum_rows, from, nrows})
  end

  def concat_r(pid, from, r_remote, iter) do
    GenServer.cast(pid, {:concat_r, from, r_remote, iter})
  end

  @impl true
  def init([x, y]) do
    type = Nx.type(x)
    {r, c} = Nx.shape(x)
    beta = Nx.dot(Nx.broadcast(0, {c, 1}), Nx.tensor(1, type: type))
    r_local = GeneralizedLinearModel.distributed_binomial_single_iter_n(x, y, beta)

    model = %GeneralizedLinearModel{
      family: Family.Binomial,
      r_local: r_local,
      coefficients: beta,
      iter: 0
    }

    initial_state = %{
      model: model,
      data: %{x: x, y: y},
      # %{iter => %{node => r_local}}
      r_remotes: %{},
      total_nrow: r,
      nodes: [],
      finished: false
    }

    {:ok, initial_state}
  end

  @impl true
  def handle_cast({:start_simulation, nodes}, state) do
    nodes =
      Enum.reduce(nodes, [], fn node, acc ->
        if node != self() do
          sum_rows(node, self(), state.total_nrow)
          [node | acc]
        else
          acc
        end
      end)

    state = %{state | nodes: nodes}

    {:noreply, state}
  end

  @impl true
  def handle_cast({:sum_rows, from, nrows}, state) do
    if Map.has_key?(state.r_remotes, from) do
      {:noreply, state}
    else
      r_remotes = Map.put_new(state.r_remotes, from, nrows)

      state =
        if length(state.nodes) == length(Map.keys(r_remotes)) do
          total_nrow =
            r_remotes
            |> Map.values()
            |> Enum.sum()
            |> Kernel.+(state.total_nrow)

          broadcast_nodes(state.nodes, state.model.r_local, state.model.iter)

          %{state | r_remotes: %{}, total_nrow: total_nrow}
        else
          %{state | r_remotes: r_remotes}
        end

      {:noreply, state}
    end
  end

  @impl true
  def handle_cast({:concat_r, _from, _r_remote, _iter}, %{finished: true} = state) do
    {:noreply, state}
  end

  @impl true
  def handle_cast({:concat_r, from, r_remote, iter}, state) do
    if Map.has_key?(state.r_remotes, iter) do
      handle_iter(from, r_remote, iter, state)
    else
      r_remotes = Map.put_new(state.r_remotes, iter, %{})
      state = %{state | r_remotes: r_remotes}

      handle_iter(from, r_remote, iter, state)
    end
  end

  defp broadcast_nodes(nodes, r_local, iter) do
    Enum.each(nodes, fn node ->
      concat_r(node, self(), r_local, iter)
    end)
  end

  defp handle_iter(from, r_remote, iter, state) do
    # state.r_remotes[iter] must not be nil
    if Map.has_key?(state.r_remotes[iter], from) do
      {:noreply, state}
    else
      iter_map = Map.put_new(state.r_remotes[iter], from, r_remote)
      r_remotes = Map.put(state.r_remotes, iter, iter_map)

      state =
        if iter == state.model.iter && length(state.nodes) == length(Map.keys(iter_map)) do
          r_local_with_all_r_remotes =
            iter_map
            |> Map.values()
            |> List.insert_at(0, state.model.r_local)
            |> Nx.concatenate()

          {r_local, beta, stop} =
            GeneralizedLinearModel.distributed_binomial_single_solve_n(
              r_local_with_all_r_remotes,
              state.model.coefficients,
              state.total_nrow,
              GeneralizedLinearModel.default_maxit(),
              GeneralizedLinearModel.default_tol(),
              state.model.iter
            )

          model = %{
            state.model
            | r_local: r_local,
              coefficients: beta,
              iter: state.model.iter + 1
          }

          finished = stop?(stop)

          if finished do
            {:ok, parent} = Registry.meta(Registry.Simulation, :parent)
            send(parent, :simulation_is_done)

            %{state | r_remotes: r_remotes, model: model, finished: true}
          else
            r_local =
              GeneralizedLinearModel.distributed_binomial_single_iter_n(
                state.data.x,
                state.data.y,
                beta
              )

            model = %{model | r_local: r_local}

            broadcast_nodes(state.nodes, model.r_local, model.iter)

            %{state | r_remotes: r_remotes, model: model}
          end
        else
          %{state | r_remotes: r_remotes}
        end

      {:noreply, state}
    end
  end

  @impl true
  def handle_call(:glm, _from, state) do
    {:reply, state.model, state}
  end

  defp stop?(stop) do
    if Nx.to_number(stop) != 0 do
      true
    else
      false
    end
  end
end
