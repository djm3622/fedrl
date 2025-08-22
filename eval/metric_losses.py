from typing import Dict
import matplotlib.pyplot as plt


def output_metric_comparison(results: Dict):
    methods = results.keys()

    first_method_key = next(iter(methods))
    if 'losses' in results[first_method_key] and 'mean' in results[first_method_key]['losses']:
         num_rounds = len(results[first_method_key]['losses']['mean'])
         rounds = range(1, num_rounds + 1)

         plt.figure(figsize=(10, 6))
         for method in methods:
             if 'losses' in results[method] and 'mean' in results[method]['losses'] and 'std' in results[method]['losses']:
                 mean_losses = results[method]['losses']['mean']
                 std_losses = results[method]['losses']['std']
                 plt.plot(rounds, mean_losses, label=f'{method} (Mean)')
                 plt.fill_between(rounds, mean_losses - std_losses, mean_losses + std_losses, alpha=0.2)

         plt.xlabel("Communication Rounds")
         plt.ylabel("Average Training Loss")
         plt.title("Average Training Loss over Communication Rounds (Mean ± Std)")
         plt.legend()
         plt.grid(True)
         plt.show()

         metrics_to_plot = []
         if 'metrics' in results[first_method_key]:
              metrics_to_plot = results[first_method_key]['metrics'].keys()

         for metric_name in metrics_to_plot:
             plt.figure(figsize=(10, 6))
             for method in methods:
                 if metric_name in results[method]['metrics'] and 'mean' in results[method]['metrics'][metric_name] and 'std' in results[method]['metrics'][metric_name]:
                      mean_metric = results[method]['metrics'][metric_name]['mean']
                      std_metric = results[method]['metrics'][metric_name]['std']
                      plt.plot(rounds, mean_metric, label=method)
                      plt.fill_between(rounds, mean_metric - std_metric, mean_metric + std_metric, alpha=0.2)

             plt.xlabel("Communication Rounds")
             plt.ylabel(metric_name)
             plt.title(f"{metric_name} over Communication Rounds (Mean ± Std)")
             plt.legend()
             plt.grid(True)
             plt.savefig(f"{metric_name}_over_rounds.png")
             plt.show()

         print("\nPlotting finished.")