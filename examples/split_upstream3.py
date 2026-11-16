"""Prepare, replay and repartition one upstream 3.x capture through public Split APIs."""

import torch
from torch import nn

from torchlens.split import SplitFeatures, prepare_graph


def main() -> None:
    """Check live weights, dynamic batches and recuts without recapturing."""

    torch.manual_seed(4)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3)).double().eval()
    graph = prepare_graph(
        model,
        torch.ones(1, 4, dtype=torch.float64),
        features=SplitFeatures(retain_trace=False),
    )
    dag = graph.compute_dag()
    runtime = graph.materialize(graph.analyze(dag.to_partition([dag.nodes[0].call_id])))
    with torch.no_grad():
        model[0].weight.add_(0.1)
        for batch in (1, 2, 4, 8):
            inputs = torch.randn(batch, 4, dtype=torch.float64)
            torch.testing.assert_close(runtime.replay(inputs), model(inputs), atol=1e-8, rtol=1e-7)
        runtime = runtime.repartition(dag.to_partition([]))
        torch.testing.assert_close(runtime.replay(inputs), model(inputs), atol=1e-8, rtol=1e-7)


if __name__ == "__main__":
    main()
