"""Statistical testing functions for experimental analysis.

This module provides functions for statistical hypothesis testing and
effect size computation to compare agent performance.
"""

from typing import List, Dict, Tuple, Optional
from pathlib import Path

import numpy as np
from scipy import stats

from .loaders import (
    LoadedTrialResult,
    filter_by_agent,
    filter_by_complexity,
    filter_successful,
    get_unique_agents,
    get_unique_complexity_levels,
)


def mannwhitney_u_test(group1: List[float], group2: List[float]) -> Dict[str, float]:
    """Perform Mann-Whitney U test (non-parametric).

    Args:
        group1: First group of measurements.
        group2: Second group of measurements.

    Returns:
        Dictionary containing:
            - statistic: U-statistic
            - p_value: Two-tailed p-value
            - significant: True if p < 0.05
    """
    if not group1 or not group2:
        return {'statistic': np.nan, 'p_value': np.nan, 'significant': False}

    statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')

    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
    }


def independent_t_test(group1: List[float], group2: List[float]) -> Dict[str, float]:
    """Perform independent samples t-test (parametric).

    Args:
        group1: First group of measurements.
        group2: Second group of measurements.

    Returns:
        Dictionary containing:
            - statistic: t-statistic
            - p_value: Two-tailed p-value
            - significant: True if p < 0.05
    """
    if not group1 or not group2:
        return {'statistic': np.nan, 'p_value': np.nan, 'significant': False}

    statistic, p_value = stats.ttest_ind(group1, group2)

    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
    }


def compute_cohens_d(group1: List[float], group2: List[float]) -> float:
    """Compute Cohen's d effect size.

    Args:
        group1: First group of measurements.
        group2: Second group of measurements.

    Returns:
        Cohen's d effect size. Interpretation:
            - Small: d = 0.2
            - Medium: d = 0.5
            - Large: d = 0.8
    """
    if not group1 or not group2:
        return np.nan

    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return np.nan

    return (mean1 - mean2) / pooled_std


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size magnitude.

    Args:
        d: Cohen's d value.

    Returns:
        Interpretation string.
    """
    if np.isnan(d):
        return "N/A"

    abs_d = abs(d)
    if abs_d < 0.2:
        return "Negligible"
    elif abs_d < 0.5:
        return "Small"
    elif abs_d < 0.8:
        return "Medium"
    else:
        return "Large"


def compare_agents(
    results: List[LoadedTrialResult],
    agent1: str,
    agent2: str,
    metric: str = 'computation_time',
    by_complexity: bool = True,
) -> Dict:
    """Compare two agents on a specific metric with statistical tests.

    Args:
        results: List of trial results.
        agent1: Name of first agent.
        agent2: Name of second agent.
        metric: Metric to compare ('computation_time', 'path_length', etc.).
        by_complexity: If True, perform separate tests for each complexity level.

    Returns:
        Dictionary containing test results.
    """
    comparison = {
        'agent1': agent1,
        'agent2': agent2,
        'metric': metric,
    }

    if by_complexity:
        complexity_levels = get_unique_complexity_levels(results)
        comparison['by_complexity'] = {}

        for level in complexity_levels:
            filtered1 = filter_by_agent(results, agent1)
            filtered1 = filter_by_complexity(filtered1, level)

            filtered2 = filter_by_agent(results, agent2)
            filtered2 = filter_by_complexity(filtered2, level)

            # Extract metric values
            if metric == 'path_length':
                filtered1 = filter_successful(filtered1)
                filtered2 = filter_successful(filtered2)

            values1 = [getattr(r, metric) for r in filtered1]
            values2 = [getattr(r, metric) for r in filtered2]

            comparison['by_complexity'][level] = {
                f'{agent1}_mean': np.mean(values1) if values1 else np.nan,
                f'{agent1}_std': np.std(values1) if values1 else np.nan,
                f'{agent2}_mean': np.mean(values2) if values2 else np.nan,
                f'{agent2}_std': np.std(values2) if values2 else np.nan,
                'mannwhitney_u': mannwhitney_u_test(values1, values2),
                't_test': independent_t_test(values1, values2),
                'cohens_d': compute_cohens_d(values1, values2),
                'effect_size_interpretation': interpret_cohens_d(compute_cohens_d(values1, values2)),
            }
    else:
        # Overall comparison
        filtered1 = filter_by_agent(results, agent1)
        filtered2 = filter_by_agent(results, agent2)

        if metric == 'path_length':
            filtered1 = filter_successful(filtered1)
            filtered2 = filter_successful(filtered2)

        values1 = [getattr(r, metric) for r in filtered1]
        values2 = [getattr(r, metric) for r in filtered2]

        comparison['overall'] = {
            f'{agent1}_mean': np.mean(values1) if values1 else np.nan,
            f'{agent1}_std': np.std(values1) if values1 else np.nan,
            f'{agent2}_mean': np.mean(values2) if values2 else np.nan,
            f'{agent2}_std': np.std(values2) if values2 else np.nan,
            'mannwhitney_u': mannwhitney_u_test(values1, values2),
            't_test': independent_t_test(values1, values2),
            'cohens_d': compute_cohens_d(values1, values2),
            'effect_size_interpretation': interpret_cohens_d(compute_cohens_d(values1, values2)),
        }

    return comparison


def generate_statistical_report(
    results: List[LoadedTrialResult],
    save_path: str,
) -> None:
    """Generate comprehensive statistical report comparing all agents.

    Args:
        results: List of trial results.
        save_path: Path to save the report text file.
    """
    agents = get_unique_agents(results)
    complexity_levels = get_unique_complexity_levels(results)

    report_lines = []
    report_lines.append("="*80)
    report_lines.append("STATISTICAL ANALYSIS REPORT")
    report_lines.append("="*80)
    report_lines.append("")

    # Success rate comparison (chi-square test)
    report_lines.append("-"*80)
    report_lines.append("SUCCESS RATE COMPARISON")
    report_lines.append("-"*80)
    for level in complexity_levels:
        report_lines.append(f"\n{level.upper()} Complexity:")
        for i, agent1 in enumerate(agents):
            for agent2 in agents[i+1:]:
                filtered1 = filter_by_agent(results, agent1)
                filtered1 = filter_by_complexity(filtered1, level)
                successes1 = sum(1 for r in filtered1 if r.success)
                total1 = len(filtered1)

                filtered2 = filter_by_agent(results, agent2)
                filtered2 = filter_by_complexity(filtered2, level)
                successes2 = sum(1 for r in filtered2 if r.success)
                total2 = len(filtered2)

                # Chi-square test (only if we have enough data)
                contingency = [[successes1, total1 - successes1],
                               [successes2, total2 - successes2]]

                # Check if contingency table has sufficient data
                if min(successes1, total1 - successes1, successes2, total2 - successes2) > 0:
                    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
                else:
                    chi2, p_value = np.nan, np.nan

                rate1 = successes1 / total1 if total1 > 0 else 0
                rate2 = successes2 / total2 if total2 > 0 else 0

                report_lines.append(f"  {agent1} vs {agent2}:")
                report_lines.append(f"    {agent1}: {successes1}/{total1} ({rate1:.1%})")
                report_lines.append(f"    {agent2}: {successes2}/{total2} ({rate2:.1%})")
                if not np.isnan(p_value):
                    report_lines.append(f"    Chi-square: χ²={chi2:.4f}, p={p_value:.4f} {'*' if p_value < 0.05 else ''}")
                else:
                    report_lines.append(f"    Chi-square: Insufficient data (0% or 100% success rate)")
                report_lines.append("")

    # Computation time comparison
    report_lines.append("-"*80)
    report_lines.append("COMPUTATION TIME COMPARISON")
    report_lines.append("-"*80)
    for i, agent1 in enumerate(agents):
        for agent2 in agents[i+1:]:
            report_lines.append(f"\n{agent1} vs {agent2}:")
            comparison = compare_agents(results, agent1, agent2, 'computation_time', by_complexity=True)

            for level in complexity_levels:
                stats_data = comparison['by_complexity'][level]
                report_lines.append(f"\n  {level.upper()}:")
                report_lines.append(f"    {agent1}: mean={stats_data[f'{agent1}_mean']:.4f}s, "
                                   f"std={stats_data[f'{agent1}_std']:.4f}s")
                report_lines.append(f"    {agent2}: mean={stats_data[f'{agent2}_mean']:.4f}s, "
                                   f"std={stats_data[f'{agent2}_std']:.4f}s")
                report_lines.append(f"    Mann-Whitney U: U={stats_data['mannwhitney_u']['statistic']:.2f}, "
                                   f"p={stats_data['mannwhitney_u']['p_value']:.4f} "
                                   f"{'*' if stats_data['mannwhitney_u']['significant'] else ''}")
                report_lines.append(f"    Independent t-test: t={stats_data['t_test']['statistic']:.4f}, "
                                   f"p={stats_data['t_test']['p_value']:.4f} "
                                   f"{'*' if stats_data['t_test']['significant'] else ''}")
                report_lines.append(f"    Cohen's d: {stats_data['cohens_d']:.4f} "
                                   f"({stats_data['effect_size_interpretation']})")

    # Path length comparison
    report_lines.append("\n" + "-"*80)
    report_lines.append("PATH LENGTH COMPARISON (Successful Trials Only)")
    report_lines.append("-"*80)
    for i, agent1 in enumerate(agents):
        for agent2 in agents[i+1:]:
            report_lines.append(f"\n{agent1} vs {agent2}:")
            comparison = compare_agents(results, agent1, agent2, 'path_length', by_complexity=True)

            for level in complexity_levels:
                stats_data = comparison['by_complexity'][level]
                report_lines.append(f"\n  {level.upper()}:")
                report_lines.append(f"    {agent1}: mean={stats_data[f'{agent1}_mean']:.2f}, "
                                   f"std={stats_data[f'{agent1}_std']:.2f}")
                report_lines.append(f"    {agent2}: mean={stats_data[f'{agent2}_mean']:.2f}, "
                                   f"std={stats_data[f'{agent2}_std']:.2f}")
                report_lines.append(f"    Mann-Whitney U: U={stats_data['mannwhitney_u']['statistic']:.2f}, "
                                   f"p={stats_data['mannwhitney_u']['p_value']:.4f} "
                                   f"{'*' if stats_data['mannwhitney_u']['significant'] else ''}")
                report_lines.append(f"    Independent t-test: t={stats_data['t_test']['statistic']:.4f}, "
                                   f"p={stats_data['t_test']['p_value']:.4f} "
                                   f"{'*' if stats_data['t_test']['significant'] else ''}")
                report_lines.append(f"    Cohen's d: {stats_data['cohens_d']:.4f} "
                                   f"({stats_data['effect_size_interpretation']})")

    report_lines.append("\n" + "="*80)
    report_lines.append("LEGEND:")
    report_lines.append("  * = Statistically significant (p < 0.05)")
    report_lines.append("  Cohen's d interpretation: <0.2=Negligible, 0.2-0.5=Small, 0.5-0.8=Medium, >0.8=Large")
    report_lines.append("="*80)

    # Write report
    with open(save_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"Statistical report saved to: {save_path}")
