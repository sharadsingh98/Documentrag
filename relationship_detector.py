class RelationshipDetector:
    def __init__(
        self,
        strong_threshold=70,
        foreign_key_threshold=30,
        weak_threshold=10,
    ):
        self.strong_threshold = strong_threshold
        self.foreign_key_threshold = foreign_key_threshold
        self.weak_threshold = weak_threshold

    @staticmethod
    def _normalize(col_name):
        return col_name.lower().strip()

    @staticmethod
    def _normalize_id(col_name):
        return col_name.lower().replace("_", "").replace(" ", "")

    def _classify_relationship(self, overlap_pct):
        if overlap_pct > self.strong_threshold:
            return "Strong Match"
        elif overlap_pct > self.foreign_key_threshold:
            return "Foreign Key"
        else:
            return "Weak Match"

    def detect(self, files_dict):
        relationships = []

        if len(files_dict) < 2:
            return relationships

        file_names = list(files_dict.keys())

        for i, file1 in enumerate(file_names):
            df1 = files_dict[file1]
            cols1 = {self._normalize(col): col for col in df1.columns}

            for file2 in file_names[i + 1:]:
                df2 = files_dict[file2]
                cols2 = {self._normalize(col): col for col in df2.columns}

                # Strategy 1: exact column name match
                common_cols = set(cols1.keys()).intersection(cols2.keys())
                for col_key in common_cols:
                    col1 = cols1[col_key]
                    col2 = cols2[col_key]

                    try:
                        values1 = set(df1[col1].dropna().astype(str).unique())
                        values2 = set(df2[col2].dropna().astype(str).unique())
                        overlap = values1.intersection(values2)

                        if not overlap:
                            continue

                        overlap_pct = (len(overlap) / max(len(values1), len(values2))) * 100

                        relationships.append({
                            "file1": file1,
                            "file2": file2,
                            "column": col1,
                            "column_file2": col2,
                            "overlap_count": len(overlap),
                            "overlap_percentage": overlap_pct,
                            "total_unique_file1": len(values1),
                            "total_unique_file2": len(values2),
                            "relationship_type": self._classify_relationship(overlap_pct),
                        })
                    except Exception:
                        continue

                # Strategy 2: ID-based match
                id_cols1 = [c for c in df1.columns if "id" in c.lower()]
                id_cols2 = [c for c in df2.columns if "id" in c.lower()]

                for col1 in id_cols1:
                    for col2 in id_cols2:
                        if self._normalize(col1) == self._normalize(col2):
                            continue

                        if self._normalize_id(col1) != self._normalize_id(col2):
                            continue

                        try:
                            values1 = set(df1[col1].dropna().astype(str).unique())
                            values2 = set(df2[col2].dropna().astype(str).unique())
                            overlap = values1.intersection(values2)

                            if not overlap:
                                continue

                            overlap_pct = (len(overlap) / max(len(values1), len(values2))) * 100
                            if overlap_pct < self.weak_threshold:
                                continue

                            relationships.append({
                                "file1": file1,
                                "file2": file2,
                                "column": col1,
                                "column_file2": col2,
                                "overlap_count": len(overlap),
                                "overlap_percentage": overlap_pct,
                                "total_unique_file1": len(values1),
                                "total_unique_file2": len(values2),
                                "relationship_type": self._classify_relationship(overlap_pct),
                            })
                        except Exception:
                            continue

        return relationships
