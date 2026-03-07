import React, { useState, useEffect } from 'react';
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
} from '@mui/material';
import {
  ModelTraining as TrainIcon,
  CheckCircle as DoneIcon,
} from '@mui/icons-material';
import { trainModel, getPreview } from '../services/api';
import type { TrainResponse, Algorithm } from '../types';

interface MLTrainingProps {
  sessionId: string;
}

const supervisedAlgos: { value: Algorithm; label: string }[] = [
  { value: 'logistic_regression', label: 'Logistic Regression' },
  { value: 'random_forest', label: 'Random Forest' },
  { value: 'knn', label: 'K-Nearest Neighbors' },
  { value: 'svm', label: 'Support Vector Machine' },
];

const unsupervisedAlgos: { value: Algorithm; label: string }[] = [
  { value: 'kmeans', label: 'K-Means Clustering' },
  { value: 'dbscan', label: 'DBSCAN' },
];

export default function MLTraining({ sessionId }: MLTrainingProps) {
  const [columns, setColumns] = useState<string[]>([]);
  const [modelType, setModelType] = useState<'supervised' | 'unsupervised'>('supervised');
  const [algorithm, setAlgorithm] = useState<Algorithm>('logistic_regression');
  const [targetColumn, setTargetColumn] = useState('');
  const [featureColumns, setFeatureColumns] = useState<string[]>([]);
  const [testSize, setTestSize] = useState('0.2');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<TrainResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getPreview(sessionId)
      .then((r) => setColumns(r.preview.columns))
      .catch(() => {});
  }, [sessionId]);

  useEffect(() => {
    setAlgorithm(modelType === 'supervised' ? 'logistic_regression' : 'kmeans');
    setTargetColumn('');
    setFeatureColumns([]);
  }, [modelType]);

  const handleTrain = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await trainModel({
        session_id: sessionId,
        model_type: modelType,
        algorithm,
        ...(modelType === 'supervised' && { target_column: targetColumn }),
        feature_columns: featureColumns,
        parameters: { test_size: parseFloat(testSize), random_state: 42 },
      });
      setResult(res);
    } catch (err: any) {
      setError(err?.response?.data?.error || err.message);
    } finally {
      setLoading(false);
    }
  };

  const algos = modelType === 'supervised' ? supervisedAlgos : unsupervisedAlgos;

  return (
    <Box sx={{ maxWidth: 900, mx: 'auto' }}>
      {/* Model type toggle */}
      <Paper elevation={0} sx={{ p: 3, mb: 3, border: '1px solid #e5e7eb', borderRadius: 2 }}>
        <Typography variant="subtitle1" fontWeight={600} sx={{ mb: 2 }}>
          Model Type
        </Typography>
        <ToggleButtonGroup
          value={modelType}
          exclusive
          onChange={(_, v) => v && setModelType(v)}
          sx={{ mb: 3 }}
        >
          <ToggleButton value="supervised" sx={{ textTransform: 'none', px: 3 }}>
            Supervised
          </ToggleButton>
          <ToggleButton value="unsupervised" sx={{ textTransform: 'none', px: 3 }}>
            Unsupervised
          </ToggleButton>
        </ToggleButtonGroup>

        <Grid container spacing={2}>
          <Grid item xs={12} sm={4}>
            <FormControl fullWidth size="small">
              <InputLabel>Algorithm</InputLabel>
              <Select
                value={algorithm}
                label="Algorithm"
                onChange={(e) => setAlgorithm(e.target.value as Algorithm)}
              >
                {algos.map((a) => (
                  <MenuItem key={a.value} value={a.value}>{a.label}</MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>

          {modelType === 'supervised' && (
            <Grid item xs={12} sm={4}>
              <FormControl fullWidth size="small">
                <InputLabel>Target Column</InputLabel>
                <Select
                  value={targetColumn}
                  label="Target Column"
                  onChange={(e) => setTargetColumn(e.target.value)}
                >
                  {columns.map((c) => (
                    <MenuItem key={c} value={c}>{c}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
          )}

          <Grid item xs={12} sm={modelType === 'supervised' ? 4 : 4}>
            <TextField
              fullWidth
              size="small"
              label="Test Size"
              type="number"
              value={testSize}
              onChange={(e) => setTestSize(e.target.value)}
              inputProps={{ min: 0.1, max: 0.5, step: 0.05 }}
            />
          </Grid>
        </Grid>

        <FormControl fullWidth size="small" sx={{ mt: 2 }}>
          <InputLabel>Feature Columns</InputLabel>
          <Select
            multiple
            value={featureColumns}
            onChange={(e) =>
              setFeatureColumns(typeof e.target.value === 'string' ? e.target.value.split(',') : e.target.value)
            }
            input={<OutlinedInput label="Feature Columns" />}
            renderValue={(sel) => (sel as string[]).join(', ')}
          >
            {columns
              .filter((c) => c !== targetColumn)
              .map((col) => (
                <MenuItem key={col} value={col}>
                  <Checkbox checked={featureColumns.includes(col)} size="small" />
                  <ListItemText primary={col} />
                </MenuItem>
              ))}
          </Select>
        </FormControl>

        <Box sx={{ mt: 3 }}>
          <Button
            variant="contained"
            startIcon={<TrainIcon />}
            onClick={handleTrain}
            disabled={loading || featureColumns.length === 0 || (modelType === 'supervised' && !targetColumn)}
            sx={{
              borderRadius: 2,
              textTransform: 'none',
              bgcolor: '#6366f1',
              '&:hover': { bgcolor: '#4f46e5' },
            }}
          >
            Train Model
          </Button>
        </Box>
      </Paper>

      {loading && <LinearProgress sx={{ borderRadius: 1, mb: 2 }} />}
      {error && <Alert severity="error" sx={{ borderRadius: 2, mb: 2 }}>{error}</Alert>}

      {/* Results */}
      {result && (
        <Paper elevation={0} sx={{ p: 3, border: '1px solid #e5e7eb', borderRadius: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
            <DoneIcon sx={{ color: '#22c55e' }} />
            <Typography variant="subtitle1" fontWeight={600}>Training Results</Typography>
            <Chip
              label={result.algorithm.replace('_', ' ')}
              size="small"
              sx={{ ml: 'auto', bgcolor: '#eef2ff', color: '#4338ca', textTransform: 'capitalize' }}
            />
          </Box>

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
                  .filter(([_, v]) => typeof v === 'number')
                  .map(([key, val]) => (
                    <TableRow key={key} hover>
                      <TableCell sx={{ textTransform: 'capitalize' }}>
                        {key.replace(/_/g, ' ')}
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={typeof val === 'number' ? val.toFixed(4) : String(val)}
                          size="small"
                          sx={{
                            bgcolor: typeof val === 'number' && val > 0.7 ? '#dcfce7' : '#fef3c7',
                            color: typeof val === 'number' && val > 0.7 ? '#166534' : '#92400e',
                            fontWeight: 600,
                          }}
                        />
                      </TableCell>
                    </TableRow>
                  ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>
      )}
    </Box>
  );
}
