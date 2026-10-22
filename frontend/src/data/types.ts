// Shapes of the JSON files written by scripts/sync-evidence.mjs.
export interface Num {
  value: number;
  source_file: string;
  pointer?: string;
  read_on: string;
}
export interface Cite {
  text: string;
  source_file: string;
  line: { value: number };
  read_on: string;
}

export interface DroughtRow {
  horizon: string;
  method: string;
  label: string;
  lookback: Num;
  published_skill: Num;
  published_rmse: Num;
  reproduced_skill: Num | null;
  reproduced_rmse: Num | null;
  identical_to_published: boolean | null;
}
export interface Drought {
  published_file: string;
  reproduced_file: string | null;
  regions: Record<string, DroughtRow[]>;
}

export interface Heat {
  manifest_file: string;
  regions: Record<string, { horizon: string; label: string; skill: Num }[]>;
  extended_search: {
    file: string;
    cells_tried: Num;
    best_cell: { region: string; target: string; feature_set: string; horizon: string; skill: Num };
    anything_clears_bar: boolean;
    read_on: string;
  };
}

export interface Checker {
  file: string;
  generated_at: string;
  what_this_measures: string;
  cases: Num;
  numbers_labelled: Num;
  true_positives: Num;
  false_positives: Num;
  false_negatives: Num;
  true_negatives: Num;
  precision: Num;
  recall: Num;
  f1: Num;
  known_defect: { text: string; source_file: string; line: { value: number }; read_on: string };
}

export interface RetrievalAgg {
  queries: Num;
  precision_at_5: Num;
  mrr: Num;
}
export interface TrackA {
  corpus_note: string;
  easy: {
    file: string;
    note: string;
    queries: Num;
    k: Num;
    precision_at_k: Num;
    mrr: Num;
    generated_at: string;
  };
  strict: {
    file: string;
    note: string;
    metric_note: string;
    pooling_assumption: string;
    generated_at: string;
    soft: Record<string, RetrievalAgg>;
    hard: Record<string, RetrievalAgg>;
  };
  default_search: {
    mode: string;
    statement: string;
    source_file: string;
    line: { value: number };
    read_on: string;
  };
}

export interface SlmCell extends Cite {
  column: string;
}
export interface SlmTable {
  header: string;
  columns: string[];
  rows: SlmCell[][];
}
export interface Slm {
  source_file: string;
  status: string;
  adapter: { id: string; url: string; base_model: string; licence: string; statement: string };
  model_quality: SlmTable;
  parity: SlmTable;
  benchmark: SlmTable;
  startup: SlmTable;
  test_split: Cite;
  limits: Cite[];
  laptop_smoke_test: Cite;
}
