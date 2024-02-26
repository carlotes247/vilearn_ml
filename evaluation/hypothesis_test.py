import argparse

from itertools import combinations
from scipy.stats import normaltest, ttest_ind, mannwhitneyu, bartlett, levene
from evaluation.eval_utils import get_metric_list


def normal_distribution(x: list) -> bool:
    """
    Test whether a sample differs from a normal distribution.
    This function tests the null hypothesis that a sample comes from a normal distribution. It is based on D’Agostino
    and Pearson’s test that combines skew and kurtosis to produce an omnibus test of normality.

    Args:
        x: A list containing the sample data

    Returns:
        A boolean,...
        if False, null hypothesis is rejected and it can be assumed that the sample come from a normal distribution
        if True, null hypothesis cannot be rejected and it can be assumed that the sample are not normal distributed
    """
    k2, p = normaltest(x)
    alpha = 0.05
    if p < alpha:  # null hypothesis: x comes from a normal distribution
        # null hypothesis can be rejected
        return False
    else:
        # null hypothesis cannot be rejected
        return True


def bartlett_test(a: list, b: list, alpha: float) -> bool:
    """
    Perform Bartlett’s test for equal variances.
    Bartlett’s test tests the null hypothesis that all input samples are from populations with equal variances.
    For samples from significantly non-normal populations, Levene’s test levene is more robust.

    Args:
        a: A list containing the sample data one
        b:  A list containing the sample data two
        alpha: A float value representing the corrected alpha value (according to Bonferroni)

    Returns:
        A boolean,...
        if False, null hypothesis is rejected and it can be assumed that the variances are equal across all samples
        if True, null hypothesis cannot be rejected and it can be assumed that the variances are not equal across
            all samples
    """
    stat, p = bartlett(a, b)
    if p < alpha:  # null hypothesis: the variances are equal across all samples/groups
        # The null hypothesis can be rejected
        return False
    else:
        # The null hypothesis cannot be rejected
        return True


def levene_test(a: list, b: list, alpha: float) -> bool:
    """
    Perform Levene test for equal variances.
    The Levene test tests the null hypothesis that all input samples are from populations with equal variances. Levene’s
    test is an alternative to Bartlett’s test bartlett in the case where there are significant deviations from normality.

    Args:
        a: A list containing the sample data one
        b:  A list containing the sample data two
        alpha: A float value representing the corrected alpha value (according to Bonferroni)

    Returns:
        A boolean,...
        if False, null hypothesis is rejected and it can be assumed that the variances are equal across all samples
        if True, null hypothesis cannot be rejected and it can be assumed that the variances are not equal across
            all samples
    """
    stat, p = levene(a, b)
    if p < alpha:  # null hypothesis: the variances are equal across all samples/groups
        # The null hypothesis can be rejected
        return False
    else:
        # The null hypothesis cannot be rejected
        return True


def student_t_test(a: list, b: list, equal_var: bool, alpha: float) -> bool:
    """
    Calculate the T-test for the means of two independent samples of scores.
    This is a test for the null hypothesis that 2 independent samples have identical average (expected) values.

    Args:
        a: A list containing the sample data one
        b:  A list containing the sample data two
        equal_var: A boolean, ...
                    if True, perform a standard independent 2 sample test that assumes equal population variances.
                    if False, perform Welch’s t-test, which does not assume equal population variance.
        alpha: A float value representing the corrected alpha value (according to Bonferroni)

    Returns:
        A boolean,...
        if False, null hypothesis is rejected and it can be assumed that the difference in the samples is
            statistically significant.
        if True, null hypothesis cannot be rejected and it can be assumed the result of a statistical coincidence.
    """
    t_stat, p = ttest_ind(a, b, axis=0, equal_var=equal_var, alternative='two-sided')
    if p < alpha:
        # Null hypothesis can be rejected"
        return False
    else:
        # Null hypothesis cannot be rejected
        return True


def mann_whitney_u_test(a: list, b: list, alpha: float) -> bool:
    """
    Perform the Mann-Whitney U rank test on two independent samples.
    The Mann-Whitney U test is a nonparametric test of the null hypothesis that the distribution underlying sample x is
    the same as the distribution underlying sample y. It is often used as a test of difference in location between
    distributions.

    Args:
        a: A list with sample data
        b: A list with sample data (different from a)
        alpha: A float representing the corrected alpha value (according to Bonferroni)

    Returns:
        A boolean, ...
        if False, null hypothesis is rejected and it can be assumed that the difference in the samples is
        statistically significant.
        if True, null hypothesis cannot be rejected and it can be assumed the result of a statistical coincidence.
    """
    U1, p = mannwhitneyu(a, b, alternative='two-sided', method="auto")

    if p < alpha:  # null hypothesis: a, b
        # The null hypothesis can be rejected"
        return False
    else:
        # The null hypothesis cannot be rejected
        return True


def significance_test(experiment_a: str, experiment_b: str, metric_key: str, alpha: float) -> None:
    """
    Perform a significance test between two experiments.

    Args:
        experiment_a: A string representing the name of the experiment a
        experiment_b: A string representing the  name of the experiment b
        metric_key: A string representing the metric of the experiments to be investigated.
        alpha: A float representing thecorrected alpha value (according to Bonferroni)

    Returns:
        None
    """
    metrics_a = get_metric_list(experiment_name=experiment_a, metric_key=metric_key)
    metrics_b = get_metric_list(experiment_name=experiment_b, metric_key=metric_key)

    assert len(metrics_a) == len(metrics_b), 'sample sizes are not equal.'

    norm_dist_a = normal_distribution(metrics_a)
    norm_dist_b = normal_distribution(metrics_b)

    if norm_dist_a and norm_dist_b:
        # T- test
        test = 'T-test'
        equal_var = bartlett_test(metrics_a, metrics_b, alpha)
        h0 = student_t_test(metrics_a, metrics_b, equal_var, alpha)
    else:
        # Mann Whitney U Test
        test = 'Mann Whitney U Test'
        h0 = mann_whitney_u_test(metrics_a, metrics_b, alpha)

    if not h0:
        print(f'{test} (alpha={round(alpha, 5)}): {experiment_a} and {experiment_b} are statistically significant.')
    else:
        print(
            f'{test} (alpha={round(alpha, 5)}): {experiment_a} and {experiment_b} are based on a statistical coincidence.')


def main(args):
    new_voice = args.new_voice

    for metric_key in ['rmse', 'log_likelihood', 'wasserstein_distance']:
        print(f'\nMetric: {metric_key}')
        if new_voice:
            experiments = ['lstm_sf_new_voice', 'csg_sf_ef_new_voice', 'lstm_sf_ef_new_voice', 'csg_sf_ef_ri_new_voice',
                           'lstm_sf_ef_ri_new_voice', 'csg_sf_ef_c_new_voice', 'lstm_sf_ef_c_new_voice']
        else:
            experiments = ['csg_sf', 'lstm_sf', 'csg_sf_ef', 'lstm_sf_ef', 'csg_sf_ef_ri', 'lstm_sf_ef_ri',
                           'csg_sf_ef_c', 'lstm_sf_ef_c']

        comb = combinations(experiments, 2)
        comb_list = [list(item) for item in comb]

        alpha = 0.05
        alpha_adj = alpha / len(comb_list)

        for c in comb_list:
            significance_test(c[0], c[1], metric_key, alpha_adj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-nv', '--new_voice', action='store_true', default=False,
                        help='Test the results of the new voice experiments.')
    args = parser.parse_args()

    main(args)
