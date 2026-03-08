import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Alert,
  Grid,
  Card,
  CardContent,
  LinearProgress,
  Chip,
  OutlinedInput,
  Checkbox,
  ListItemText,
  TextField,
  ToggleButtonGroup,
  ToggleButton,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Collapse,
  IconButton,
  Divider,
  Tooltip,
} from '@mui/material';
import {
  ModelTraining as TrainIcon,
  CheckCircle as DoneIcon,
  Replay as RetryIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  Delete as DeleteIcon,
  History as HistoryIcon,
  ErrorOutline as ErrorIcon,
  Refresh as RunAgainIcon,
  Tune as LoadConfigIcon,
} from '@mui/icons-material';
import { trainModel, getPreview, warmupML } from '../services/api';
import type { TrainResponse, Algorithm } from '../types';

interface MLTrainingProps {
  sessionId: string;
}

// ── Per-run log entry saved to localStorage ────────────────────────────────
interface MLRunEntry {
  id: string;
  timestamp: string;
  modelType: 'supervised' | 'unsupervised';
  algorithm: Algorithm;
  targetColumn: string;
  featureColumns: string[];
  scaler: 'none' | 'standard' | 'minmax';
  testSize: number;
  datasetPref: 'processed' | 'original'; // user's chosen preference
  metrics?: Record<string, number>;
  status: 'success' | 'error';
  errorMessage?: string;
  datasetType?: 'processed' | 'original'; // what was actually used
}

// Config snapshot used to support Re-run / Retry without relying on state
interface TrainConfig {
  modelType: 'supervised' | 'unsupervised';
  algorithm: Algorithm;
  targetColumn: string;
  featureColumns: string[];
  scaler: 'none' | 'standard' | 'minmax';
  testSize: string;
  datasetPref: 'processed' | 'original';
}

const supervisedAlgos: { value: Algorithm; label: string; isRegression?: boolean }[] = [
  { value: 'logistic_regression', label: 'Logistic Regression' },
  { value: 'linear_regression', label: 'Linear Regression', isRegression: true },
  { value: 'random_forest', label: 'Random Forest' },
  { value: 'knn', label: 'K-Nearest Neighbors' },
  { value: 'svm', label: 'Support Vector Machine' },
];

const unsupervisedAlgos: { value: Algorithm; label: string }[] = [
  { value: 'kmeans', label: 'K-Means Clustering' },
  { value: 'dbscan', label: 'DBSCAN' },
];

const ALGO_COLORS: Record<string, { bgcolor: string; color: string }> = {
  logistic_regression: { bgcolor: '#eff6ff', color: '#1d4ed8' },
  linear_regression:   { bgcolor: '#f0fdf4', color: '#15803d' },
  random_forest:       { bgcolor: '#fefce8', color: '#854d0e' },
  knn:                 { bgcolor: '#fdf4ff', color: '#7e22ce' },
  svm:                 { bgcolor: '#fff7ed', color: '#c2410c' },
  kmeans:              { bgcolor: '#ecfdf5', color: '#065f46' },
  dbscan:              { bgcolor: '#f8fafc', color: '#475569' },
};

function algoLabel(algo: string) {
  return (
    supervisedAlgos.find((a) => a.value === algo)?.label ||
    unsupervisedAlgos.find((a) => a.value === algo)?.label ||
    algo.replace(/_/g, ' ')
  );
}

function isRegressionAlgo(a: Algorithm) { return a === 'linear_regression'; }

/** Primary headline metric for a run entry (shown in collapsed header). */
function primaryMetric(
  metrics: Record<string, number>,
  modelType: string,
  algo: Algorithm,
): [string, number] | null {
  if (modelType === 'supervised') {
    if (isRegressionAlgo(algo) && metrics.r2_score !== undefined) return ['R²', metrics.r2_score];
    if (metrics.accuracy !== undefined) return ['Accuracy', metrics.accuracy];
  }
  if (metrics.silhouette_score !== undefined) return ['Silhouette', metrics.silhouette_score];
  if (metrics.n_clusters !== undefined) return ['Clusters', metrics.n_clusters];
  return null;
}

// ── Log-entry card ─────────────────────────────────────────────────────────
function LogEntryCard({
  entry,
  onRerun,
  onLoadConfig,
  onDelete,
  isRunning,
}: {
  entry: MLRunEntry;
  onRerun: () => void;
  onLoadConfig: () => void;
  onDelete: () => void;
  isRunning: boolean;
}) {
  const [expanded, setExpanded] = useState(false);
  const color = ALGO_COLORS[entry.algorithm] ?? { bgcolor: '#f3f4f6', color: '#111827' };
  const pm = entry.metrics ? primaryMetric(entry.metrics, entry.modelType, entry.algorithm) : null;
  const timeStr = new Date(entry.timestamp).toLocaleString(undefined, {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  });
  const isSupervised = entry.modelType === 'supervised';

  return (
    <Paper
      elevation={0}
      sx={{
        border: `1px solid ${entry.status === 'error' ? '#fecaca' : '#e5e7eb'}`,
        borderLeft: `4px solid ${entry.status === 'error' ? '#ef4444' : '#22c55e'}`,
        borderRadius: 2,
        mb: 1.5,
        p: 2,
      }}
    >
      {/* Header */}
      <Box sx={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 1 }}>
        <Chip label={algoLabel(entry.algorithm)} size="small" sx={{ ...color, fontWeight: 700, height: 22 }} />
        <Chip label={entry.modelType} size="small" variant="outlined" sx={{ height: 20, fontSize: 10, textTransform: 'capitalize' }} />
        {entry.status === 'error' && (
          <Chip
            icon={<ErrorIcon sx={{ fontSize: '13px !important' }} />}
            label="Failed"
            size="small"
            sx={{ bgcolor: '#fef2f2', color: '#b91c1c', height: 22 }}
          />
        )}
        {pm && entry.status === 'success' && (
          <Chip
            label={`${pm[0]}: ${typeof pm[1] === 'number' ? pm[1].toFixed(4) : pm[1]}`}
            size="small"
            sx={{
              bgcolor: pm[1] > 0.7 ? '#dcfce7' : '#fef3c7',
              color: pm[1] > 0.7 ? '#15803d' : '#854d0e',
              fontWeight: 700, height: 22,
            }}
          />
        )}
        <Typography variant="caption" color="text.secondary" sx={{ ml: 'auto', mr: 0.5, fontSize: 11 }}>
          {timeStr}
        </Typography>
        <Tooltip title="Re-run with this exact config">
          <span>
            <IconButton size="small" onClick={onRerun} disabled={isRunning} sx={{ color: '#6366f1' }}>
              <RetryIcon fontSize="small" />
            </IconButton>
          </span>
        </Tooltip>
        <Tooltip title="Load config into form">
          <span>
            <IconButton size="small" onClick={onLoadConfig} disabled={isRunning} sx={{ color: '#0284c7' }}>
              <LoadConfigIcon fontSize="small" />
            </IconButton>
          </span>
        </Tooltip>
        <Tooltip title="Delete from log">
          <IconButton size="small" onClick={onDelete} sx={{ color: '#9ca3af' }}>
            <DeleteIcon fontSize="small" />
          </IconButton>
        </Tooltip>
        <IconButton size="small" onClick={() => setExpanded((e) => !e)}>
          {expanded ? <ExpandLessIcon fontSize="small" /> : <ExpandMoreIcon fontSize="small" />}
        </IconButton>
      </Box>

      {/* Collapsed summary line */}
      {!expanded && (
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1.5, mt: 0.75 }}>
          {isSupervised && entry.targetColumn && (
            <Typography variant="caption" color="text.secondary">
              Y: <strong>{entry.targetColumn}</strong>
            </Typography>
          )}
          <Typography variant="caption" color="text.secondary">
            X: <strong>
              {entry.featureColumns.slice(0, 3).join(', ')}
              {entry.featureColumns.length > 3 ? ` +${entry.featureColumns.length - 3} more` : ''}
            </strong>
          </Typography>
          {entry.status === 'error' && entry.errorMessage && (
            <Typography variant="caption" sx={{ color: '#b91c1c' }}>
              {entry.errorMessage.slice(0, 70)}{entry.errorMessage.length > 70 ? '…' : ''}
            </Typography>
          )}
        </Box>
      )}

      {/* Expanded detail */}
      <Collapse in={expanded}>
        <Box sx={{ mt: 1.5 }}>
          {/* X & Y labels */}
          <Grid container spacing={1} sx={{ mb: 1 }}>
            {isSupervised && entry.targetColumn && (
              <Grid item xs={12}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Typography variant="caption" fontWeight={700} sx={{ minWidth: 80 }}>Y Target:</Typography>
                  <Chip label={entry.targetColumn} size="small" sx={{ bgcolor: '#fdf2f8', color: '#7c3aed', height: 20, fontWeight: 600 }} />
                </Box>
              </Grid>
            )}
            <Grid item xs={12}>
              <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1 }}>
                <Typography variant="caption" fontWeight={700} sx={{ minWidth: 80, mt: 0.25 }}>X Features:</Typography>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                  {entry.featureColumns.map((c) => (
                    <Chip key={c} label={c} size="small" sx={{ bgcolor: '#eff6ff', color: '#1d4ed8', height: 20, fontSize: 11 }} />
                  ))}
                </Box>
              </Box>
            </Grid>
            <Grid item xs={12}>
              <Box sx={{ display: 'flex', gap: 2 }}>
                <Typography variant="caption" color="text.secondary">
                  Test size: <strong>{(entry.testSize * 100).toFixed(0)}%</strong>
                </Typography>
                {entry.scaler !== 'none' && (
                  <Typography variant="caption" color="text.secondary">
                    Scaler: <strong>{entry.scaler}</strong>
                  </Typography>
                )}
              </Box>
            </Grid>
          </Grid>

          {entry.status === 'error' && entry.errorMessage && (
            <Alert severity="error" sx={{ py: 0.5, fontSize: 12, mb: 1 }}>{entry.errorMessage}</Alert>
          )}

          {entry.datasetType && (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
              <Typography variant="caption" fontWeight={700} sx={{ minWidth: 80 }}>Data used:</Typography>
              <Chip
                label={entry.datasetType === 'processed' ? 'Processed data' : 'Original data'}
                size="small"
                sx={{
                  bgcolor: entry.datasetType === 'processed' ? '#f0fdf4' : '#fefce8',
                  color: entry.datasetType === 'processed' ? '#15803d' : '#854d0e',
                  height: 20, fontWeight: 600, fontSize: 11,
                }}
              />
            </Box>
          )}

          {entry.metrics && entry.status === 'success' && (
            <>
              <Divider sx={{ my: 1 }} />
              <Typography variant="caption" fontWeight={700} color="text.secondary" sx={{ display: 'block', mb: 0.75 }}>
                Metrics
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                {Object.entries(entry.metrics)
                  .filter(([, v]) => typeof v === 'number')
                  .map(([k, v]) => {
                    const isGood = (k === 'accuracy' || k === 'r2_score' || k === 'f1_score' || k === 'silhouette_score') && v > 0.7;
                    return (
                      <Box key={k} sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                        <Typography variant="caption" color="text.secondary" sx={{ fontSize: 11 }}>
                          {k.replace(/_/g, ' ')}:
                        </Typography>
                        <Chip
                          label={v.toFixed(4)}
                          size="small"
                          sx={{ height: 20, fontSize: 11, fontWeight: 700, bgcolor: isGood ? '#dcfce7' : '#f3f4f6', color: isGood ? '#15803d' : '#374151' }}
                        />
                      </Box>
                    );
                  })}
              </Box>
            </>
          )}
        </Box>
      </Collapse>
    </Paper>
  );
}

// ── Main component ─────────────────────────────────────────────────────────
export default function MLTraining({ sessionId }: MLTrainingProps) {
  const [columns, setColumns] = useState<string[]>([]);
  const [modelType, setModelType] = useState<'supervised' | 'unsupervised'>('supervised');
  const [algorithm, setAlgorithm] = useState<Algorithm>('logistic_regression');
  const [targetColumn, setTargetColumn] = useState('');
  const [featureColumns, setFeatureColumns] = useState<string[]>([]);
  const [testSize, setTestSize] = useState('0.2');
  const [scaler, setScaler] = useState<'none' | 'standard' | 'minmax'>('none');
  const [datasetPref, setDatasetPref] = useState<'processed' | 'original'>('processed');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<TrainResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Config captured at each train call, used by Retry button
  const [lastConfig, setLastConfig] = useState<TrainConfig | null>(null);
  // warmupAttempt counts retries so the chip shows progress
  const [warmupAttempt, setWarmupAttempt] = useState(0);
  const [warmupStatus, setWarmupStatus] = useState<'idle' | 'warming' | 'ready'>('idle');
  // 'pinging' = first probe; 'coldstart' = 504 detected, waiting for bootstrap
  const [warmupPhase, setWarmupPhase] = useState<'pinging' | 'coldstart'>('pinging');

  // Run log — persisted per session in localStorage
  const [runLog, setRunLog] = useState<MLRunEntry[]>(() => {
    try {
      const saved = localStorage.getItem(`mlRunLog_${sessionId}`);
      if (!saved) return [];
      const parsed = JSON.parse(saved);
      // Validate it's an array of objects with the expected shape
      if (!Array.isArray(parsed)) return [];
      return parsed.filter(
        (e: any) => e && typeof e === 'object' && typeof e.id === 'string' && typeof e.algorithm === 'string'
      ) as MLRunEntry[];
    } catch { return []; }
  });
  useEffect(() => {
    try { localStorage.setItem(`mlRunLog_${sessionId}`, JSON.stringify(runLog)); } catch {}
  }, [runLog, sessionId]);

  // Prevent the modelType effect from resetting algo/cols when loading a config
  const skipModelTypeReset = useRef(false);

  useEffect(() => {
    getPreview(sessionId).then((r) => setColumns(r.preview.columns)).catch(() => {});
    // Retry-loop warmup: keeps pinging until the Lambda cold-start finishes (can take >29s)
    setWarmupStatus('warming');
    setWarmupAttempt(0);
    setWarmupPhase('pinging');
    warmupML(
      (n) => setWarmupAttempt(n),
      () => setWarmupPhase('coldstart'),
    ).finally(() => setWarmupStatus('ready'));
  }, [sessionId]);

  useEffect(() => {
    if (skipModelTypeReset.current) { skipModelTypeReset.current = false; return; }
    setAlgorithm(modelType === 'supervised' ? 'logistic_regression' : 'kmeans');
    setTargetColumn('');
    setFeatureColumns([]);
    setScaler('none');
  }, [modelType]);

  const algos = modelType === 'supervised' ? supervisedAlgos : unsupervisedAlgos;
  const isRegression = isRegressionAlgo(algorithm);

  // ── Core train function — accepts an override config for re-run / retry ──
  const handleTrain = async (cfg?: TrainConfig) => {
    const mt  = cfg?.modelType      ?? modelType;
    const algo = (cfg?.algorithm    ?? algorithm) as Algorithm;
    const tc  = cfg?.targetColumn   ?? targetColumn;
    const fc  = cfg?.featureColumns ?? featureColumns;
    const sc  = cfg?.scaler         ?? scaler;
    const ts  = cfg?.testSize       ?? testSize;
    const dp  = cfg?.datasetPref    ?? datasetPref;

    const captured: TrainConfig = { modelType: mt, algorithm: algo, targetColumn: tc, featureColumns: fc, scaler: sc, testSize: ts, datasetPref: dp };
    setLastConfig(captured);
    setLoading(true);
    setError(null);
    setResult(null);

    const entryId = `${Date.now()}`;
    const timestamp = new Date().toISOString();

    try {
      const res = await trainModel({
        session_id: sessionId,
        model_type: mt,
        algorithm: algo,
        ...(mt === 'supervised' && { target_column: tc }),
        feature_columns: fc,
        dataset_type: dp,
        parameters: { test_size: parseFloat(ts), random_state: 42, scaler: sc },
      });
      setResult(res);
      setWarmupStatus('ready');

      // Extract numeric metrics for the log
      const metrics: Record<string, number> = {};
      if (res.results?.metrics) {
        Object.entries(res.results.metrics).forEach(([k, v]) => {
          if (typeof v === 'number') metrics[k] = v;
        });
      }
      setRunLog((prev) => [{
        id: entryId, timestamp, modelType: mt, algorithm: algo,
        targetColumn: tc, featureColumns: fc, scaler: sc,
        testSize: parseFloat(ts), datasetPref: dp, metrics, status: 'success',
        datasetType: res.dataset_type,
      }, ...prev]);
    } catch (err: any) {
      const status  = err?.response?.status;
      const rawMsg  = err?.response?.data?.error || err?.response?.data?.message || err?.message || 'Unknown error';
      // "Network Error" = no HTTP response → almost always a cold-start CORS/timeout issue
      const isNetworkErr = !err.response && (err.message === 'Network Error' || err.code === 'ERR_NETWORK');
      const isTimeout = status === 504 || isNetworkErr ||
        rawMsg.toLowerCase().includes('timed out') ||
        rawMsg.toLowerCase().includes('endpoint request');
      const displayMsg = isTimeout
        ? 'The ML server is still warming up (cold start can take ~2 min while ML packages load). Please click Retry in a moment.'
        : rawMsg;
      setError(displayMsg);
      setRunLog((prev) => [{
        id: entryId, timestamp, modelType: mt, algorithm: algo,
        targetColumn: tc, featureColumns: fc, scaler: sc,
        testSize: parseFloat(ts), datasetPref: dp, status: 'error', errorMessage: displayMsg,
      }, ...prev]);
    } finally {
      setLoading(false);
    }
  };

  const handleRerun = (entry: MLRunEntry) => {
    handleTrain({
      modelType: entry.modelType, algorithm: entry.algorithm,
      targetColumn: entry.targetColumn, featureColumns: entry.featureColumns,
      scaler: entry.scaler, testSize: String(entry.testSize),
      datasetPref: entry.datasetPref ?? 'processed',
    });
  };

  const handleLoadConfig = (entry: MLRunEntry) => {
    // Suppress the modelType effect so it doesn't wipe algo / cols
    skipModelTypeReset.current = true;
    setModelType(entry.modelType);
    setAlgorithm(entry.algorithm);
    setTargetColumn(entry.targetColumn);
    setFeatureColumns(entry.featureColumns);
    setScaler(entry.scaler);
    setTestSize(String(entry.testSize));
    setDatasetPref(entry.datasetPref ?? 'processed');
    setResult(null);
    setError(null);
  };

  return (
    <Box sx={{ maxWidth: 900, mx: 'auto' }}>

      {/* ── Config panel ── */}
      <Paper elevation={0} sx={{ p: 3, mb: 3, border: '1px solid #e5e7eb', borderRadius: 2 }}>
        <Typography variant="subtitle1" fontWeight={600} sx={{ mb: 2 }}>Model Type</Typography>
        <ToggleButtonGroup
          value={modelType}
          exclusive
          onChange={(_, v) => v && setModelType(v)}
          sx={{ mb: 3 }}
        >
          <ToggleButton value="supervised" sx={{ textTransform: 'none', px: 3 }}>Supervised</ToggleButton>
          <ToggleButton value="unsupervised" sx={{ textTransform: 'none', px: 3 }}>Unsupervised</ToggleButton>
        </ToggleButtonGroup>

        <Grid container spacing={2}>
          <Grid item xs={12} sm={4}>
            <FormControl fullWidth size="small">
              <InputLabel>Algorithm</InputLabel>
              <Select value={algorithm} label="Algorithm" onChange={(e) => setAlgorithm(e.target.value as Algorithm)}>
                {algos.map((a) => <MenuItem key={a.value} value={a.value}>{a.label}</MenuItem>)}
              </Select>
            </FormControl>
          </Grid>

          {modelType === 'supervised' && (
            <Grid item xs={12} sm={4}>
              <FormControl fullWidth size="small">
                <InputLabel>Target Column (Y)</InputLabel>
                <Select value={targetColumn} label="Target Column (Y)" onChange={(e) => setTargetColumn(e.target.value)}>
                  {columns.map((c) => <MenuItem key={c} value={c}>{c}</MenuItem>)}
                </Select>
              </FormControl>
            </Grid>
          )}

          <Grid item xs={12} sm={4}>
            <TextField
              fullWidth size="small" label="Test Size" type="number"
              value={testSize} onChange={(e) => setTestSize(e.target.value)}
              inputProps={{ min: 0.1, max: 0.5, step: 0.05 }}
            />
          </Grid>
        </Grid>

        {/* Feature scaling */}
        {modelType === 'supervised' && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="body2" fontWeight={600} sx={{ mb: 1, color: '#374151' }}>Feature Scaling</Typography>
            <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap' }}>
              {[
                { value: 'none', label: 'No Scaler', sub: 'Use raw features' },
                { value: 'standard', label: 'Standard Scaler', sub: 'Mean=0, Std=1' },
                { value: 'minmax', label: 'Min-Max Scaler', sub: 'Values in [0, 1]' },
              ].map((s) => (
                <Card
                  key={s.value} elevation={0}
                  onClick={() => setScaler(s.value as 'none' | 'standard' | 'minmax')}
                  sx={{
                    cursor: 'pointer',
                    border: scaler === s.value ? '2px solid #6366f1' : '1px solid #e5e7eb',
                    borderRadius: 2, bgcolor: scaler === s.value ? '#eef2ff' : '#fff',
                    px: 2, py: 1, minWidth: 140, transition: 'all 0.15s',
                    '&:hover': { borderColor: '#6366f1' },
                  }}
                >
                  <CardContent sx={{ p: '8px !important' }}>
                    <Typography variant="body2" fontWeight={600} sx={{ color: scaler === s.value ? '#4338ca' : '#111827' }}>
                      {s.label}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">{s.sub}</Typography>
                  </CardContent>
                </Card>
              ))}
            </Box>
          </Box>
        )}

        {/* Dataset selection */}
        <Box sx={{ mt: 2 }}>
          <Typography variant="body2" fontWeight={600} sx={{ mb: 1, color: '#374151' }}>Training Data</Typography>
          <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap' }}>
            {[
              { value: 'processed', label: 'Processed Data', sub: 'Use preprocessed dataset', color: '#15803d', bg: '#f0fdf4', border: '#bbf7d0' },
              { value: 'original',  label: 'Original Data',  sub: 'Use raw uploaded dataset', color: '#854d0e', bg: '#fefce8', border: '#fde68a' },
            ].map((d) => (
              <Card
                key={d.value} elevation={0}
                onClick={() => setDatasetPref(d.value as 'processed' | 'original')}
                sx={{
                  cursor: 'pointer',
                  border: datasetPref === d.value ? `2px solid ${d.border}` : '1px solid #e5e7eb',
                  borderRadius: 2, bgcolor: datasetPref === d.value ? d.bg : '#fff',
                  px: 2, py: 1, minWidth: 160, transition: 'all 0.15s',
                  '&:hover': { borderColor: d.border },
                }}
              >
                <CardContent sx={{ p: '8px !important' }}>
                  <Typography variant="body2" fontWeight={600} sx={{ color: datasetPref === d.value ? d.color : '#111827' }}>
                    {d.label}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">{d.sub}</Typography>
                </CardContent>
              </Card>
            ))}
          </Box>
        </Box>

        {/* Feature columns (X) */}
        <FormControl fullWidth size="small" sx={{ mt: 2 }}>
          <InputLabel>Feature Columns (X)</InputLabel>
          <Select
            multiple value={featureColumns}
            onChange={(e) => setFeatureColumns(typeof e.target.value === 'string' ? e.target.value.split(',') : e.target.value as string[])}
            input={<OutlinedInput label="Feature Columns (X)" />}
            renderValue={(sel) => (sel as string[]).join(', ')}
          >
            {columns.filter((c) => c !== targetColumn).map((col) => (
              <MenuItem key={col} value={col}>
                <Checkbox checked={featureColumns.includes(col)} size="small" />
                <ListItemText primary={col} />
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        <Box sx={{ mt: 3, display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
          <Button
            variant="contained"
            startIcon={<TrainIcon />}
            onClick={() => handleTrain()}
            disabled={loading || featureColumns.length === 0 || (modelType === 'supervised' && !targetColumn)}
            sx={{ borderRadius: 2, textTransform: 'none', bgcolor: '#6366f1', '&:hover': { bgcolor: '#4f46e5' } }}
          >
            Train Model
          </Button>
          {warmupStatus === 'warming' && (
            <Chip
              label={
                warmupPhase === 'coldstart'
                  ? warmupAttempt <= 1
                    ? 'Cold start – loading ML packages (~2 min)…'
                    : `Checking if ready… (${warmupAttempt - 1})`
                  : 'Connecting to ML server…'
              }
              size="small"
              sx={{ bgcolor: '#fefce8', color: '#854d0e', fontWeight: 600, fontSize: 11,
                    border: '1px solid #fde68a' }}
            />
          )}
          {warmupStatus === 'ready' && (
            <Chip label="Server ready ✓" size="small" sx={{ bgcolor: '#f0fdf4', color: '#15803d', fontWeight: 600, fontSize: 11, border: '1px solid #bbf7d0' }} />
          )}
        </Box>
      </Paper>

      {loading && <LinearProgress sx={{ borderRadius: 1, mb: 2 }} />}

      {/* Error with Retry button */}
      {error && (
        <Alert
          severity="error"
          sx={{ borderRadius: 2, mb: 2 }}
          action={
            <Button
              size="small"
              startIcon={<RetryIcon />}
              onClick={() => handleTrain(lastConfig ?? undefined)}
              disabled={loading}
              sx={{ textTransform: 'none', fontWeight: 600, whiteSpace: 'nowrap' }}
            >
              Retry
            </Button>
          }
        >
          {error}
        </Alert>
      )}

      {/* ── Current result ── */}
      {result && (
        <Paper elevation={0} sx={{ p: 3, border: '1px solid #e5e7eb', borderRadius: 2, mb: 3 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2, flexWrap: 'wrap' }}>
            <DoneIcon sx={{ color: '#22c55e' }} />
            <Typography variant="subtitle1" fontWeight={600}>Training Results</Typography>
            <Chip
              label={algoLabel(result.algorithm)}
              size="small"
              sx={{ ...(ALGO_COLORS[result.algorithm] ?? { bgcolor: '#f3f4f6', color: '#111827' }), fontWeight: 700 }}
            />
            {result.dataset_type && (
              <Chip
                label={result.dataset_type === 'processed' ? 'Processed data' : 'Original data'}
                size="small"
                sx={{
                  bgcolor: result.dataset_type === 'processed' ? '#f0fdf4' : '#fefce8',
                  color: result.dataset_type === 'processed' ? '#15803d' : '#854d0e',
                  fontWeight: 700, border: '1px solid',
                  borderColor: result.dataset_type === 'processed' ? '#bbf7d0' : '#fde68a',
                }}
              />
            )}
            <Box sx={{ ml: 'auto', display: 'flex', gap: 1 }}>
              <Button
                size="small" variant="outlined" startIcon={<RunAgainIcon />}
                onClick={() => handleTrain()} disabled={loading}
                sx={{ textTransform: 'none', borderRadius: 2, borderColor: '#6366f1', color: '#6366f1' }}
              >
                Run Again
              </Button>
            </Box>
          </Box>

          {/* Y target */}
          {result.results?.target_column && (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
              <Typography variant="caption" fontWeight={700} sx={{ minWidth: 80 }}>Y Target:</Typography>
              <Chip label={result.results.target_column} size="small" sx={{ bgcolor: '#fdf2f8', color: '#7c3aed', height: 22, fontWeight: 600 }} />
            </Box>
          )}

          {/* X features */}
          {result.results?.feature_columns?.length > 0 && (
            <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1, mb: 2 }}>
              <Typography variant="caption" fontWeight={700} sx={{ minWidth: 80, mt: 0.25 }}>X Features:</Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                {result.results.feature_columns.map((c) => (
                  <Chip key={c} label={c} size="small" sx={{ bgcolor: '#eff6ff', color: '#1d4ed8', height: 20, fontSize: 11 }} />
                ))}
              </Box>
            </Box>
          )}

          {/* Metrics table */}
          <TableContainer>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell sx={{ fontWeight: 700, bgcolor: '#f9fafb' }}>Metric</TableCell>
                  <TableCell sx={{ fontWeight: 700, bgcolor: '#f9fafb' }}>Value</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {Object.entries(result.results.metrics)
                  .filter(([, v]) => typeof v === 'number')
                  .map(([key, val]) => {
                    const numVal = typeof val === 'number' ? val : Number(val);
                    const isGoodKey = key === 'r2_score' || key === 'accuracy' || key.includes('score');
                    const isGood = isGoodKey && numVal > 0.7 && !(isRegression && key !== 'r2_score');
                    return (
                      <TableRow key={key} hover>
                        <TableCell sx={{ textTransform: 'capitalize' }}>{key.replace(/_/g, ' ')}</TableCell>
                        <TableCell>
                          <Chip
                            label={numVal.toFixed(4)}
                            size="small"
                            sx={{ bgcolor: isGood ? '#dcfce7' : '#fef3c7', color: isGood ? '#166534' : '#92400e', fontWeight: 600 }}
                          />
                        </TableCell>
                      </TableRow>
                    );
                  })}
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>
      )}

      {/* ── Training history log ── */}
      {runLog.length > 0 && (
        <Paper elevation={0} sx={{ p: 3, border: '1px solid #e5e7eb', borderRadius: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
            <HistoryIcon sx={{ color: '#6366f1', fontSize: 20 }} />
            <Typography variant="subtitle1" fontWeight={600}>Training History</Typography>
            <Chip
              label={`${runLog.length} run${runLog.length !== 1 ? 's' : ''}`}
              size="small"
              sx={{ bgcolor: '#eef2ff', color: '#4338ca' }}
            />
            <Button
              size="small"
              onClick={() => setRunLog([])}
              sx={{ ml: 'auto', color: '#9ca3af', textTransform: 'none' }}
            >
              Clear All
            </Button>
          </Box>

          {runLog.map((entry) => (
            <LogEntryCard
              key={entry.id}
              entry={entry}
              onRerun={() => handleRerun(entry)}
              onLoadConfig={() => handleLoadConfig(entry)}
              onDelete={() => setRunLog((prev) => prev.filter((e) => e.id !== entry.id))}
              isRunning={loading}
            />
          ))}
        </Paper>
      )}
    </Box>
  );
}
