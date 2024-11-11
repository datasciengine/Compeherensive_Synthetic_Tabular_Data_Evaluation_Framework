class LDiversityChecker:
    def __init__(self):
        pass

    @staticmethod
    def check_l_diversity(df, quasi_identifiers, sensitive_cols, l_range=(2, 5)):
        result_dict = {}
        for l in range(l_range[0], l_range[1]):
            groups = df.groupby(quasi_identifiers)
            for name, group in groups:
                for sensitive_value in sensitive_cols:
                    if group[sensitive_value].nunique() < l:
                        result_dict[f"{l}"] = False
            result_dict[f"{l}"] = True
        return result_dict
