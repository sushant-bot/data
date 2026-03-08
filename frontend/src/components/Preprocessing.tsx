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
  Chip,
  Alert,
  Grid,
  Card,
  CardContent,
  IconButton,
  LinearProgress,
  Switch,
  FormControlLabel,
  OutlinedInput,
  Checkbox,
  ListItemText,
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  PlayArrow as RunIcon,
  CheckCircle as DoneIcon,
  Download as DownloadIcon,
} from '@mui/icons-material';
import { preprocess, getPreview } from '../services/api';
import type { PreprocessOperation, PreprocessResponse } from '../types';

interface PreprocessingProps {
  sessionId: string;
}

interface OpConfig {
  operation: string;
  method: string;
  columns: string[];
  remove: boolean;
}

const defaultOp = (): OpConfig => ({
  operation: 'handle_missing',
  method: 'fill',
  columns: [],
  remove: false,
});

export default function Preprocessing({ sessionId }: PreprocessingProps) {
  const [allColumns, setAllColumns] = useState<string[]>([]);
  const [ops, setOps] = useState<OpConfig[]>([defaultOp()]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PreprocessResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getPreview(sessionId)
      .then((r) => setAllColumns(r.preview.columns))
      .catch(() => {});
  }, [sessionId]);

  const updateOp = (i: number, patch: Partial<OpConfig>) => {
    setOps((prev) => prev.map((o, idx) => (idx === i ? { ...o, ...patch } : o)));
  };

  const removeOp = (i: number) => setOps((prev) => prev.filter((_, idx) => idx !== i));

  const handleRun = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const operations: PreprocessOperation[] = ops.map((o) => {
        if (o.operation === 'handle_missing')
          return { operation: 'handle_missing', method: o.method as 'fill' | 'drop', columns: o.columns };
        if (o.operation === 'detect_outliers')
          return { operation: 'detect_outliers', method: o.method as 'iqr' | 'zscore', remove: o.remove };
        if (o.operation === 'scale_features')
          return { operation: 'scale_features', method: o.method as 'standard' | 'minmax', columns: o.columns };
        return { operation: 'encode_categorical', method: o.method as 'label' | 'onehot', columns: o.columns };
      });
      const res = await preprocess({ session_id: sessionId, operations });
      setResult(res);
    } catch (err: any) {
      setError(err?.response?.data?.error || err.message);
    } finally {
      setLoading(false);
    }
  };

  const opOptions = [
    { value: 'handle_missing', label: 'Handle Missing Values' },
    { value: 'detect_outliers', label: 'Detect Outliers' },
    { value: 'scale_features', label: 'Scale Features' },
    { value: 'encode_categorical', label: 'Encode Categorical' },
  ];

  const methodsFor = (op: string) => {
    switch (op) {
      case 'handle_missing': return [{ value: 'fill', label: 'Fill' }, { value: 'drop', label: 'Drop' }];
      case 'detect_outliers': return [{ value: 'iqr', label: 'IQR' }, { value: 'zscore', label: 'Z-Score' }];
      case 'scale_features': return [{ value: 'standard', label: 'Standard' }, { value: 'minmax', label: 'Min-Max' }];
      case 'encode_categorical': return [{ value: 'label', label: 'Label' }, { value: 'onehot', label: 'One-Hot' }];
      default: return [];
    }
  };

  const needsColumns = (op: string) => op !== 'detect_outliers';

  return (
    <Box sx={{ maxWidth: 900, mx: 'auto' }}>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Configure preprocessing operations to clean and transform your dataset.
      </Typography>

      {ops.map((op, i) => (
        <Card
          key={i}
          elevation={0}
          sx={{ mb: 2, border: '1px solid #e5e7eb', borderRadius: 2 }}
        >
          <CardContent>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Chip label={`Step ${i + 1}`} size="small" sx={{ bgcolor: '#eef2ff', color: '#4338ca' }} />
              {ops.length > 1 && (
                <IconButton size="small" onClick={() => removeOp(i)} color="error">
                  <DeleteIcon fontSize="small" />
                </IconButton>
              )}
            </Box>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={4}>
                <FormControl fullWidth size="small">
                  <InputLabel>Operation</InputLabel>
                  <Select
                    value={op.operation}
                    label="Operation"
                    onChange={(e) =>
                      updateOp(i, {
                        operation: e.target.value,
                        method: methodsFor(e.target.value)[0]?.value || '',
                        columns: [],
                      })
                    }
                  >
                    {opOptions.map((o) => (
                      <MenuItem key={o.value} value={o.value}>{o.label}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} sm={4}>
                <FormControl fullWidth size="small">
                  <InputLabel>Method</InputLabel>
                  <Select
                    value={op.method}
                    label="Method"
                    onChange={(e) => updateOp(i, { method: e.target.value })}
                  >
                    {methodsFor(op.operation).map((m) => (
                      <MenuItem key={m.value} value={m.value}>{m.label}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              {needsColumns(op.operation) ? (
                <Grid item xs={12} sm={4}>
                  <FormControl fullWidth size="small">
                    <InputLabel>Columns</InputLabel>
                    <Select
                      multiple
                      value={op.columns}
                      onChange={(e) =>
                        updateOp(i, { columns: typeof e.target.value === 'string' ? e.target.value.split(',') : e.target.value })
                      }
                      input={<OutlinedInput label="Columns" />}
                      renderValue={(selected) => (selected as string[]).join(', ')}
                    >
                      {allColumns.map((col) => (
                        <MenuItem key={col} value={col}>
                          <Checkbox checked={op.columns.includes(col)} size="small" />
                          <ListItemText primary={col} />
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>
              ) : (
                <Grid item xs={12} sm={4}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={op.remove}
                        onChange={(e) => updateOp(i, { remove: e.target.checked })}
                      />
                    }
                    label="Remove outliers"
                  />
                </Grid>
              )}
            </Grid>
          </CardContent>
        </Card>
      ))}

      <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
        <Button
          variant="outlined"
          startIcon={<AddIcon />}
          onClick={() => setOps((prev) => [...prev, defaultOp()])}
          sx={{ borderRadius: 2, textTransform: 'none' }}
        >
          Add Step
        </Button>
        <Button
          variant="contained"
          startIcon={<RunIcon />}
          onClick={handleRun}
          disabled={loading}
          sx={{
            borderRadius: 2,
            textTransform: 'none',
            bgcolor: '#6366f1',
            '&:hover': { bgcolor: '#4f46e5' },
          }}
        >
          Run Preprocessing
        </Button>
      </Box>

      {loading && <LinearProgress sx={{ borderRadius: 1, mb: 2 }} />}
      {error && <Alert severity="error" sx={{ borderRadius: 2 }}>{error}</Alert>}
      {result && (
        <Box>
          <Alert icon={<DoneIcon />} severity="success" sx={{ borderRadius: 2, mb: 2 }}>
            Preprocessing complete — {result.operations_applied ?? result.operations_completed ?? 0} operation(s) applied.
          </Alert>
          {result.download_url && (
            <Button
              variant="outlined"
              startIcon={<DownloadIcon />}
              component="a"
              href={result.download_url}
              download="processed_data.csv"
              sx={{
                borderRadius: 2,
                textTransform: 'none',
                borderColor: '#6366f1',
                color: '#6366f1',
                '&:hover': { borderColor: '#4f46e5', bgcolor: '#eef2ff' },
              }}
            >
              Download Processed Data
            </Button>
          )}
        </Box>
      )}
    </Box>
  );
}
