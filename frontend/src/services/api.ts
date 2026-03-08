import axios from 'axios';
import type {
  UploadResponse,
  PreviewResponse,
  PreprocessRequest,
  PreprocessResponse,
  QualityResponse,
  TrainRequest,
  TrainResponse,
  TrainStartResponse,
  TrainStatusResponse,
  VisualizationRequest,
  VisualizationResponse,
  RecommendationsResponse,
  Session,
} from '../types';

const API_BASE_URL = process.env.REACT_APP_API_URL || '/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: { 'Content-Type': 'application/json' },
});

// Upload
export async function uploadFile(file: File): Promise<UploadResponse> {
  const content = await fileToBase64(file);
  const { data } = await api.post<UploadResponse>('/upload', {
    file_content: content,
    file_name: file.name,
  });
  return data;
}

// Preview
export async function getPreview(sessionId: string): Promise<PreviewResponse> {
  const { data } = await api.get<PreviewResponse>(`/preview/${encodeURIComponent(sessionId)}`);
  return data;
}

// Preprocess
export async function preprocess(req: PreprocessRequest): Promise<PreprocessResponse> {
  const { data } = await api.post<PreprocessResponse>('/preprocess', req);
  return data;
}

// Quality
export async function getQuality(sessionId: string): Promise<QualityResponse> {
  const { data } = await api.get<QualityResponse>(`/quality/${encodeURIComponent(sessionId)}`);
  return data;
}

// Train (async pattern)
export async function startTraining(req: TrainRequest): Promise<TrainStartResponse> {
  const { data } = await api.post<TrainStartResponse>('/train', req);
  return data;
}

export async function checkTrainingStatus(sessionId: string, operationId: string): Promise<TrainStatusResponse> {
  const { data } = await api.post<TrainStatusResponse>('/train', {
    action: 'check_status',
    session_id: sessionId,
    operation_id: operationId,
  });
  return data;
}

// Train (legacy sync - kept for backwards compatibility)
export async function trainModel(req: TrainRequest): Promise<TrainResponse> {
  const { data } = await api.post<TrainResponse>('/train', req);
  return data;
}

// Visualizations
export async function generateVisualization(req: VisualizationRequest): Promise<VisualizationResponse> {
  const { data } = await api.post<VisualizationResponse>(
    `/visualizations/${encodeURIComponent(req.session_id)}`,
    req
  );
  return data;
}

// AI Recommendations
export async function getRecommendations(sessionId: string): Promise<RecommendationsResponse> {
  const { data } = await api.get<RecommendationsResponse>(`/recommendations/${encodeURIComponent(sessionId)}`);
  return data;
}

// Session
export async function getSession(sessionId: string): Promise<Session> {
  const { data } = await api.get<Session>(`/sessions/${encodeURIComponent(sessionId)}`);
  return data;
}

// Helpers
function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      resolve(result.split(',')[1]); // strip data URL prefix
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}
