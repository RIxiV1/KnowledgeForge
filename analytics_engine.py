import pandas as pd
import os
import re

def is_analytic_question(question):
    """
    Detect if a question is asking for data analytics/aggregation.

    Matches whole words, not substrings, so ordinary questions aren't misrouted
    to the spreadsheet path -- e.g. "summarize" must not trigger on the "sum"
    inside it, "meaning" must not trigger on "mean".
    """
    q = question.lower()
    single_word = {
        "count", "total", "sum", "average", "avg", "mean", "highest", "lowest",
        "most", "least", "top", "bottom", "maximum", "minimum", "spent",
        "revenue", "profit", "percentage", "compare", "trend", "distribution",
        "calculate", "rank", "sort", "rows", "records", "entries", "unique",
    }
    phrases = ["group by", "how many", "per year", "per month"]
    words = set(re.findall(r"[a-z]+", q))
    return bool(words & single_word) or any(p in q for p in phrases)


def analyze_dataframe(file_path, question):
    """
    Main function to analyze dataframe - pattern matching only
    No LLM code generation (too error prone)
    """
    try:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".csv":
            df = pd.read_csv(file_path)
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)
        else:
            return "Unsupported file format for analytics"

        # Validate dataframe
        if df.empty:
            return "The dataset is empty"
        q = question.lower()
        
        if any(word in q for word in ["how many", "count", "total", "rows", "records", "entries"]):
            if "unique" in q:
                for col in df.columns:
                    if col.lower() in q:
                        unique_count = df[col].nunique()
                        return f"Unique {col}: {unique_count:,}"
                return f"Unique values in {df.columns[0]}: {df[df.columns[0]].nunique():,}"
            else:
                return f"Total Records: {len(df):,}"
            
        if any(word in q for word in ["sum", "total", "all together"]):
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                for col in numeric_cols:
                    if any(word in q for word in [col.lower(), "price", "amount", "cost", "revenue", "sales", "value"]):
                        total = df[col].sum()
                        return f"Total {col}: {total:,.2f}"
                total = df[numeric_cols[0]].sum()
                return f"Total {numeric_cols[0]}: {total:,.2f}"
            return "No numeric columns found"

# AVERAGE / MEAN
        if any(word in q for word in ["average", "avg", "mean"]):
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                for col in numeric_cols:
                    if col.lower() in q or any(word in q for word in [col.lower(), "price", "amount", "cost"]):
                        avg = df[col].mean()
                        return f"Average {col}: {avg:,.2f}"
                result = df[numeric_cols].mean()
                return result.to_string()
            return "No numeric columns found"

        if any(word in q for word in ["maximum", "max", "highest", "largest"]):
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                for col in numeric_cols:
                    if col.lower() in q or any(word in q for word in [col.lower(), "price", "amount", "value"]):
                        max_val = df[col].max()
                        return f"Maximum {col}: {max_val:,.2f}"
                result = df[numeric_cols].max()
                return result.to_string()
            return "No numeric columns found"

# MINIMUM 
        if any(word in q for word in ["minimum", "min", "lowest", "smallest"]):
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                for col in numeric_cols:
                    if col.lower() in q or any(word in q for word in [col.lower(), "price", "amount", "value"]):
                        min_val = df[col].min()
                        return f"Minimum {col}: {min_val:,.2f}"
                result = df[numeric_cols].min()
                return result.to_string()
            return "No numeric columns found"

# TOP / BOTTOM
        if "top" in q or "bottom" in q:
            match = re.search(r'(\d+)', q)
            n = int(match.group(1)) if match else 5
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                col = numeric_cols[0]
                for c in numeric_cols:
                    if c.lower() in q:
                        col = c
                        break
                if "top" in q or "highest" in q:
                    result = df.nlargest(n, col)[col].to_string()
                    return f"Top {n} {col}:\n{result}"
                else:
                    result = df.nsmallest(n, col)[col].to_string()
                    return f"Bottom {n} {col}:\n{result}"
            return "No numeric columns found"
# GROUP BY
        if "by" in q or "per" in q or "group" in q:
            # find group column
            for col in df.columns:
                if col.lower() in q:
                    # Found a column name in question
                    result = df[col].value_counts().to_string()
                    return f"Breakdown by {col}:\n{result}"
            return "Could not identify column for grouping"
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            stats = df[numeric_cols].describe().to_string()
            return f"Statistics:\n{stats}"
        
        return f"Dataset has {len(df):,} records with columns: {', '.join(df.columns.tolist())}"

    except Exception as e:
        return f"Error analyzing data: {str(e)}"
