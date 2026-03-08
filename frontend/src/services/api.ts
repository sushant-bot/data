import axios from 'axios';
import type {
  UploadResponse,
  PreviewResponse,
  PreprocessRequest,
  PreprocessResponse,
  QualityResponse,
  TrainRequest,
  TrainResponse,
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
export async function getPreview(
  sessionId: string,
  page = 1,
  pageSize = 15,
  datasetType: 'original' | 'processed' = 'original',
): Promise<PreviewResponse> {
  const { data } = await api.get<PreviewResponse>(
    `/preview/${encodeURIComponent(sessionId)}`,
    { params: { page, page_size: pageSize, dataset_type: datasetType } }
  );
  return data;
}

// Download URL
export async function getDownloadUrl(
  sessionId: string,
  dataset: 'original' | 'processed',
  format: 'csv' | 'json' = 'csv',
): Promise<string> {
  const { data } = await api.get<{ download_url: string }>(
    `/preview/${encodeURIComponent(sessionId)}`,
    { params: { download: dataset, format } }
  );
  return data.download_url;
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

// Train — long timeout; API Gateway hard-caps at 29 s but we let it respond naturally
export async function trainModel(req: TrainRequest): Promise<TrainResponse> {
  const { data } = await api.post<TrainResponse>('/train', req, { timeout: 120_000 });
  return data;
}

// Warmup — sends one probe to trigger the cold start, then waits for it to finish.
//
// The Lambda downloads ~100 MB of ML packages on first boot (60-120s), which always
// exceeds API Gateway's 29s hard timeout on the first call.
// Spamming retries every few seconds just spawns extra cold containers downloading
// in parallel. Instead: one probe → wait 90s → check every 15s.
export async function warmupML(
  onAttempt?: (n: number) => void,
  onColdStart?: () => void,
): Promise<void> {
  // Probe 1 — instant if Lambda is already warm
  onAttempt?.(1);
  try {
    await api.post('/train', { warmup: true }, { timeout: 28_000 });
    return; // was warm
  } catch {
    // Cold start triggered — Lambda is now bootstrapping (downloading ML packages).
    // Don't send more requests yet; they'd only spin up extra cold containers.
    onColdStart?.();
  }

  // Wait ~90s for bootstrap to complete, then poll every 15s.
  await new Promise(r => setTimeout(r, 90_000));
  for (let i = 0; i < 6; i++) {
    onAttempt?.(i + 2);
    try {
      await api.post('/train', { warmup: true }, { timeout: 10_000 });
      return; // warm now
    } catch {
      if (i < 5) await new Promise(r => setTimeout(r, 15_000));
    }
  }
  // Give up — user can still try training directly
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
