// === Session ===
export interface WorkflowState {
  upload_completed: boolean;
  preprocessing_completed: boolean;
  quality_assessed: boolean;
  model_trained: boolean;
  recommendations_generated: boolean;
}

export interface Session {
  session_id: string;
  timestamp: string;
  dataset_name: string;
  status: 'uploaded' | 'processed' | 'trained' | 'completed';
  workflow_state: WorkflowState;
}

// === Upload ===
export interface UploadResponse {
  session_id: string;
  dataset_name: string;
  statistics: {
    row_count: number;
    column_count: number;
    data_types: Record<string, string>;
    missing_values: Record<string, number>;
  };
  pii_detection: {
    pii_detected: boolean;
    pii_details: Record<string, string[]>;
  };
}

// === Preview ===
export interface PreviewResponse {
  session_id: string;
  dataset_name: string;
  preview: {
    columns: string[];
    rows: any[][];
    total_rows_shown: number;
    total_rows_available: number;
  };
  statistics: {
    column_stats: Record<string, any>;
    data_distribution: Record<string, any>;
  };
  metadata: {
    total_rows: number;
    total_columns: number;
    file_size: number;
    upload_timestamp: string;
  };
}

// === Preprocessing ===
export type PreprocessOperation =
  | { operation: 'handle_missing'; method: 'fill' | 'drop'; columns: string[] }
  | { operation: 'detect_outliers'; method: 'iqr' | 'zscore'; remove: boolean }
  | { operation: 'scale_features'; method: 'standard' | 'minmax'; columns: string[] }
  | { operation: 'encode_categorical'; method: 'label' | 'onehot'; columns: string[] };

export interface PreprocessRequest {
  session_id: string;
  operations: PreprocessOperation[];
}

export interface PreprocessResponse {
  session_id: string;
  operations_applied?: number;
  operations_completed?: number;
  processed_dataset_location?: string;
  download_url?: string;
  operation_results: any[];
}

// === Quality ===
export interface QualityReport {
  basic_metrics: {
    total_rows: number;
    total_columns: number;
    total_cells: number;
    memory_usage_mb: number;
  };
  missing_value_analysis: {
    total_missing_values: number;
    total_missing_percentage: number;
    columns_with_missing: number;
    missing_per_column: Record<string, number>;
    high_missing_columns: string[];
    moderate_missing_columns: string[];
    complete_columns: string[];
  };
  duplicate_analysis: {
    duplicate_rows: number;
    duplicate_percentage: number;
    unique_rows: number;
    has_duplicates: boolean;
  };
  data_imbalance_analysis: {
    categorical_columns_count: number;
    overall_imbalance_ratio: number;
    severely_imbalanced_columns: string[];
    moderately_imbalanced_columns: string[];
  };
  data_type_analysis: {
    column_type_analysis: Record<string, any>;
    type_distribution: Record<string, number>;
    numeric_columns: string[];
    categorical_columns: string[];
    potential_conversions: string[];
  };
  outlier_analysis: {
    numeric_columns_analyzed: number;
    columns_with_outliers: Record<string, any>;
  };
  overall_quality_score: number;
  recommendations: string[];
}

export interface QualityResponse {
  session_id: string;
  dataset_name: string;
  dataset_type: 'original' | 'processed';
  quality_report: QualityReport;
  timestamp: string;
}

// === ML Training ===
export type Algorithm =
  | 'logistic_regression'
  | 'random_forest'
  | 'knn'
  | 'svm'
  | 'kmeans'
  | 'dbscan';

export interface TrainRequest {
  session_id: string;
  model_type: 'supervised' | 'unsupervised';
  algorithm: Algorithm;
  target_column?: string;
  feature_columns: string[];
  parameters?: Record<string, any>;
}

export interface TrainResponse {
  session_id: string;
  model_type: string;
  algorithm: string;
  results: {
    metrics: Record<string, any>;
    training_details: Record<string, any>;
    model_location: string;
  };
}

// === Visualization ===
export type VisualizationType =
  | 'correlation_heatmap'
  | 'confusion_matrix'
  | 'roc_curve'
  | 'cluster_plot'
  | 'feature_importance';

export interface VisualizationRequest {
  session_id: string;
  visualization_type: VisualizationType;
  parameters?: {
    dataset_type?: 'processed' | 'original';
    columns?: string[];
    prediction_column?: string;
  };
}

export interface VisualizationResponse {
  session_id: string;
  visualization_type: string;
  visualization_key: string;
  presigned_url: string;
}

// === AI Recommendations ===
export interface RecommendationsResponse {
  session_id: string;
  recommendations: {
    preprocessing_suggestions: string[];
    model_suggestions: string[];
    feature_engineering_ideas: string[];
    warnings: string[];
    quality_recommendations: string[];
    quality_score: number;
  };
  data_characteristics: {
    num_rows: number;
    num_columns: number;
    num_numeric: number;
    num_categorical: number;
    missing_percentage: number;
    duplicate_percentage: number;
  };
  cached: boolean;
}
