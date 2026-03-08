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
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Divider,
  Collapse,
  Tooltip,
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  PlayArrow as RunIcon,
  CheckCircle as DoneIcon,
  ErrorOutline as FailIcon,
  ExpandMore,
  ExpandLess,
  TrendingDown,
  TrendingUp,
  Undo as UndoIcon,
  Redo as RedoIcon,
} from '@mui/icons-material';
import { preprocess, getPreview } from '../services/api';
import type { PreprocessOperation, PreprocessResponse, OperationResult } from '../types';

interface PreprocessingProps {
  sessionId: string;
  onProcessed?: () => void;
}

interface OpConfig {
  operation: string;
  method: string;
  columns: string[];
  remove: boolean;
}

// An entry in the persistent operations log
interface LogEntry {
  id: string;                // unique local id
  opConfig: OpConfig;        // original config used to build this operation
  result: OperationResult;   // stats from the server response
  applied: boolean;          // true = applied in current processed data, false = undone
}

const defaultOp = (): OpConfig => ({
  operation: 'handle_missing',
  method: 'fill',
  columns: [],
  remove: false,
});

// ── Operation result card (used inside the log) ────────────────────────────────
function OperationResultCard({
  entry,
  index,
  onUndo,
  onRedo,
  onDelete,
  undoDisabled,
  redoDisabled,
}: {
  entry: LogEntry;
  index: number;
  onUndo: () => void;
  onRedo: () => void;
  onDelete: () => void;
  undoDisabled: boolean;
  redoDisabled: boolean;
}) {
  const [expanded, setExpanded] = useState(true);
  const { result, applied } = entry;

  const opLabels: Record<string, string> = {
    null_removal: 'Handle Missing — Drop',
    null_filling: 'Handle Missing — Fill',
    outlier_removal: 'Outlier Detection & Removal',
    scaling: 'Feature Scaling',
    label_encoding: 'Label Encoding',
    one_hot_encoding: 'One-Hot Encoding',
  };
  const label = opLabels[result.action] || result.original_operation || result.action || `Step ${index + 1}`;
  const success = result.status === 'completed';

  const rowsChanged = result.rows_after - result.rows_before;
  const colsChanged = result.columns_after - result.columns_before;

  const statRows: { label: string; value: React.ReactNode }[] = [];
  statRows.push({ label: 'Rows before', value: result.rows_before?.toLocaleString() });
  statRows.push({ label: 'Rows after', value: result.rows_after?.toLocaleString() });
  if (rowsChanged !== 0) {
    statRows.push({
      label: 'Rows changed',
      value: (
        <Chip
          icon={rowsChanged < 0 ? <TrendingDown fontSize="small" /> : <TrendingUp fontSize="small" />}
          label={`${rowsChanged > 0 ? '+' : ''}${rowsChanged.toLocaleString()}`}
          size="small"
          sx={{ bgcolor: rowsChanged < 0 ? '#dcfce7' : '#fef2f2', color: rowsChanged < 0 ? '#15803d' : '#b91c1c', height: 22, fontWeight: 600 }}
        />
      ),
    });
  }
  if (colsChanged !== 0) {
    statRows.push({ label: 'Columns changed', value: <Chip label={`${colsChanged > 0 ? '+' : ''}${colsChanged} cols`} size="small" sx={{ bgcolor: '#eff6ff', color: '#1d4ed8', height: 22, fontWeight: 600 }} /> });
  }
  if (result.nulls_before !== undefined) statRows.push({ label: 'Null values before', value: result.nulls_before.toLocaleString() });
  if (result.nulls_after !== undefined) {
    statRows.push({
      label: 'Null values after',
      value: (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <span>{result.nulls_after.toLocaleString()}</span>
          {(result.nulls_before ?? 0) > (result.nulls_after ?? 0) && (
            <Chip label={`−${((result.nulls_before ?? 0) - result.nulls_after).toLocaleString()} nulls`} size="small" sx={{ bgcolor: '#dcfce7', color: '#15803d', height: 20, fontWeight: 600 }} />
          )}
        </Box>
      ),
    });
  }
  if (result.total_nulls_filled !== undefined && result.total_nulls_filled > 0) statRows.push({ label: 'Null values filled', value: result.total_nulls_filled.toLocaleString() });
  if (result.rows_removed !== undefined && result.rows_removed > 0) statRows.push({ label: 'Rows removed', value: result.rows_removed.toLocaleString() });
  if (result.outliers_removed !== undefined) statRows.push({ label: 'Outliers removed', value: result.outliers_removed.toLocaleString() });
  if (result.strategy_used) statRows.push({ label: 'Strategy', value: result.strategy_used });
  if (result.method_used) statRows.push({ label: 'Method', value: result.method_used });

  const colsAffected = Array.isArray(result.columns_affected)
    ? result.columns_affected
    : result.columns_affected && result.columns_affected !== 'all'
    ? [result.columns_affected]
    : [];

  return (
    <Card
      elevation={0}
      sx={{
        border: `1px solid ${!applied ? '#d1d5db' : success ? '#bbf7d0' : '#fecaca'}`,
        borderLeft: `4px solid ${!applied ? '#9ca3af' : success ? '#22c55e' : '#ef4444'}`,
        borderRadius: 2,
        mb: 2,
        opacity: applied ? 1 : 0.65,
        transition: 'opacity 0.2s, border-color 0.2s',
      }}
    >
      <CardContent sx={{ pb: '12px !important' }}>
        {/* Header */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: expanded ? 1.5 : 0 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
            {success ? <DoneIcon sx={{ color: applied ? '#22c55e' : '#9ca3af', fontSize: 20 }} /> : <FailIcon sx={{ color: '#ef4444', fontSize: 20 }} />}
            <Typography variant="body2" fontWeight={700}>{label}</Typography>
            <Chip label={`Step ${index + 1}`} size="small" sx={{ bgcolor: '#eef2ff', color: '#4338ca', height: 20, fontSize: 11 }} />
            {/* Applied / Not Applied badge */}
            <Chip
              label={applied ? 'Applied' : 'Not Applied'}
              size="small"
              sx={{
                height: 20,
                fontSize: 11,
                fontWeight: 700,
                bgcolor: applied ? '#dcfce7' : '#f3f4f6',
                color: applied ? '#15803d' : '#6b7280',
              }}
            />
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            {/* Undo button: only shown if applied */}
            {applied && (
              <Tooltip title="Undo this operation (revert processed data)">
                <span>
                  <IconButton size="small" onClick={onUndo} disabled={undoDisabled} color="warning">
                    <UndoIcon fontSize="small" />
                  </IconButton>
                </span>
              </Tooltip>
            )}
            {/* Redo button: only shown if not applied */}
            {!applied && (
              <Tooltip title="Redo this operation (re-apply to processed data)">
                <span>
                  <IconButton size="small" onClick={onRedo} disabled={redoDisabled} color="primary">
                    <RedoIcon fontSize="small" />
                  </IconButton>
                </span>
              </Tooltip>
            )}
            <Tooltip title="Remove from log">
              <IconButton size="small" onClick={onDelete} color="error">
                <DeleteIcon fontSize="small" />
              </IconButton>
            </Tooltip>
            <IconButton size="small" onClick={() => setExpanded((e) => !e)}>
              {expanded ? <ExpandLess fontSize="small" /> : <ExpandMore fontSize="small" />}
            </IconButton>
          </Box>
        </Box>

        <Collapse in={expanded}>
          {result.error && <Alert severity="error" sx={{ mb: 1.5, py: 0.5 }}>{result.error}</Alert>}

          <Grid container spacing={1} sx={{ mb: colsAffected.length > 0 ? 1.5 : 0 }}>
            {statRows.map((s) => (
              <Grid item xs={12} sm={6} md={4} key={s.label}>
                <Paper elevation={0} sx={{ bgcolor: '#f9fafb', borderRadius: 1, px: 1.5, py: 1, border: '1px solid #e5e7eb' }}>
                  <Typography variant="caption" color="text.secondary" display="block">{s.label}</Typography>
                  <Box sx={{ fontWeight: 600, fontSize: 13 }}>{s.value}</Box>
                </Paper>
              </Grid>
            ))}
          </Grid>

          {colsAffected.length > 0 && (
            <Box>
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                Columns affected ({colsAffected.length})
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                {colsAffected.map((c: string) => (
                  <Chip key={c} label={c} size="small" sx={{ bgcolor: '#f0f9ff', color: '#0369a1', height: 20, fontSize: 11 }} />
                ))}
              </Box>
            </Box>
          )}

          {result.null_per_column_before && expanded && (() => {
            const changed = Object.entries(result.null_per_column_before as Record<string, number>)
              .filter(([col, before]) => before > 0 || ((result.null_per_column_after as any)?.[col] ?? 0) > 0)
              .slice(0, 10);
            if (changed.length === 0) return null;
            return (
              <Box sx={{ mt: 1.5 }}>
                <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                  Null counts by column (top {changed.length})
                </Typography>
                <TableContainer component={Paper} elevation={0} sx={{ border: '1px solid #e5e7eb', borderRadius: 1 }}>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell sx={{ fontWeight: 700, bgcolor: '#f9fafb', py: 0.5 }}>Column</TableCell>
                        <TableCell sx={{ fontWeight: 700, bgcolor: '#f0fdf4', py: 0.5, color: '#15803d' }}>Before</TableCell>
                        <TableCell sx={{ fontWeight: 700, bgcolor: '#eff6ff', py: 0.5, color: '#1d4ed8' }}>After</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {changed.map(([col, before]) => {
                        const after = (result.null_per_column_after as any)?.[col] ?? 0;
                        return (
                          <TableRow key={col} hover>
                            <TableCell sx={{ py: 0.5 }}>{col}</TableCell>
                            <TableCell sx={{ bgcolor: '#f0fdf4', py: 0.5 }}>{(before as number).toLocaleString()}</TableCell>
                            <TableCell sx={{ bgcolor: '#eff6ff', py: 0.5 }}>
                              {after.toLocaleString()}
                              {(before as number) > after && (
                                <Chip label={`−${((before as number) - after).toLocaleString()}`} size="small" sx={{ ml: 1, bgcolor: '#dcfce7', color: '#15803d', height: 18, fontSize: 10 }} />
                              )}
                            </TableCell>
                          </TableRow>
                        );
                      })}
                    </TableBody>
                  </Table>
                </TableContainer>
              </Box>
            );
          })()}
        </Collapse>
      </CardContent>
    </Card>
  );
}

// Helper: convert OpConfig → PreprocessOperation for API
function opConfigToApiOp(o: OpConfig): PreprocessOperation {
  if (o.operation === 'handle_missing')
    return { operation: 'handle_missing', method: o.method as 'fill' | 'drop', columns: o.columns };
  if (o.operation === 'detect_outliers')
    return { operation: 'detect_outliers', method: o.method as 'iqr' | 'zscore', remove: o.remove };
  if (o.operation === 'scale_features')
    return { operation: 'scale_features', method: o.method as 'standard' | 'minmax', columns: o.columns };
  return { operation: 'encode_categorical', method: o.method as 'label' | 'onehot', columns: o.columns };
}

// ── Main Preprocessing component ───────────────────────────────────────────────
export default function Preprocessing({ sessionId, onProcessed }: PreprocessingProps) {
  const [allColumns, setAllColumns] = useState<string[]>([]);
  const [ops, setOps] = useState<OpConfig[]>([defaultOp()]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Persistent operation log — stored in localStorage keyed by sessionId so it
  // survives navigation between tabs/routes.
  const [log, setLog] = useState<LogEntry[]>(() => {
    try {
      const saved = localStorage.getItem(`preprocessLog_${sessionId}`);
      if (!saved) return [];
      const parsed = JSON.parse(saved);
      if (!Array.isArray(parsed)) return [];
      // Validate each entry has the minimum expected shape; drop corrupt ones
      return parsed.filter(
        (e: any) => e && typeof e === 'object' && typeof e.id === 'string' && e.opConfig && e.result
      ) as LogEntry[];
    } catch { return []; }
  });

  // Keep localStorage in sync whenever the log changes
  useEffect(() => {
    try { localStorage.setItem(`preprocessLog_${sessionId}`, JSON.stringify(log)); } catch {}
  }, [log, sessionId]);

  // True while undo/redo re-run is in progress
  const [rerunLoading, setRerunLoading] = useState(false);

  useEffect(() => {
    getPreview(sessionId)
      .then((r) => setAllColumns(r.preview.columns))
      .catch(() => {});
  }, [sessionId]);

  const updateOp = (i: number, patch: Partial<OpConfig>) => {
    setOps((prev) => prev.map((o, idx) => (idx === i ? { ...o, ...patch } : o)));
  };

  const removeOp = (i: number) => setOps((prev) => prev.filter((_, idx) => idx !== i));

  // Run all currently-queued ops and append results to the log
  const handleRun = async () => {
    setLoading(true);
    setError(null);
    try {
      const newOps = ops.map(opConfigToApiOp);

      // Include ALL previously-applied log entries FIRST so processed.csv always
      // reflects every applied step, not just the latest batch.
      const prevOps = log
        .filter((e) => e.applied)
        .map((e) => opConfigToApiOp(e.opConfig));

      const res = await preprocess({ session_id: sessionId, operations: [...prevOps, ...newOps] });

      // Only the last newOps.length results belong to the newly added steps
      const newResults = res.operation_results.slice(-newOps.length);

      const newEntries: LogEntry[] = newOps.map((_, i) => ({
        id: `${Date.now()}-${i}`,
        opConfig: ops[i] ?? ops[ops.length - 1],
        result: (newResults[i] ?? res.operation_results[res.operation_results.length - 1]) as OperationResult,
        applied: true,
      }));
      setLog((prev) => [...prev, ...newEntries]);

      if (res.processed_dataset?.s3_key) {
        onProcessed?.();
      }
    } catch (err: any) {
      setError(err?.response?.data?.error || err.message);
    } finally {
      setLoading(false);
    }
  };

  // Re-run only the "applied" entries from the log (used for undo/redo)
  const rerunApplied = async (entries: LogEntry[]) => {
    setRerunLoading(true);
    try {
      const appliedOps = entries.filter((e) => e.applied).map((e) => opConfigToApiOp(e.opConfig));
      if (appliedOps.length === 0) {
        // Nothing applied — we can't easily "reset" to original via the API.
        // Best we can do: inform user.
        setError('All operations undone. The processed data currently reflects the last server state. Re-run to fully reset.');
        setRerunLoading(false);
        return;
      }
      await preprocess({ session_id: sessionId, operations: appliedOps });
      onProcessed?.();
    } catch (err: any) {
      setError(err?.response?.data?.error || err.message);
    } finally {
      setRerunLoading(false);
    }
  };

  const handleUndo = (id: string) => {
    setLog((prev) => {
      const updated = prev.map((e) => (e.id === id ? { ...e, applied: false } : e));
      // Async: re-run all still-applied entries
      rerunApplied(updated);
      return updated;
    });
  };

  const handleRedo = (id: string) => {
    setLog((prev) => {
      const updated = prev.map((e) => (e.id === id ? { ...e, applied: true } : e));
      rerunApplied(updated);
      return updated;
    });
  };

  const handleDelete = (id: string) => {
    setLog((prev) => {
      const updated = prev.filter((e) => e.id !== id);
      // If deleted entry was applied, re-run remaining applied
      const wasApplied = prev.find((e) => e.id === id)?.applied ?? false;
      if (wasApplied) rerunApplied(updated);
      return updated;
    });
  };

  const opOptions = [
    { value: 'handle_missing', label: 'Handle Missing Values' },
    { value: 'detect_outliers', label: 'Detect / Remove Outliers' },
    { value: 'scale_features', label: 'Scale Features' },
    { value: 'encode_categorical', label: 'Encode Categorical' },
  ];

  const methodsFor = (op: string) => {
    switch (op) {
      case 'handle_missing': return [{ value: 'fill', label: 'Fill (mean/mode)' }, { value: 'drop', label: 'Drop Rows' }];
      case 'detect_outliers': return [{ value: 'iqr', label: 'IQR' }, { value: 'zscore', label: 'Z-Score' }];
      case 'scale_features': return [{ value: 'standard', label: 'Standard Scaler' }, { value: 'minmax', label: 'Min-Max Scaler' }];
      case 'encode_categorical': return [{ value: 'label', label: 'Label Encoding' }, { value: 'onehot', label: 'One-Hot Encoding' }];
      default: return [];
    }
  };

  const needsColumns = (op: string) => op !== 'detect_outliers';

  const appliedCount = log.filter((e) => e.applied && e.result.status === 'completed').length;

  return (
    <Box sx={{ maxWidth: 960, mx: 'auto' }}>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Configure preprocessing operations below. Results are logged — you can undo, redo, or delete individual steps.
      </Typography>

      {/* ── Operation builder ── */}
      {ops.map((op, i) => (
        <Card key={i} elevation={0} sx={{ mb: 2, border: '1px solid #e5e7eb', borderRadius: 2 }}>
          <CardContent>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Chip label={`Step ${i + 1}`} size="small" sx={{ bgcolor: '#eef2ff', color: '#4338ca' }} />
                <Typography variant="body2" color="text.secondary">{opOptions.find(o => o.value === op.operation)?.label}</Typography>
              </Box>
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
                    onChange={(e) => updateOp(i, { operation: e.target.value, method: methodsFor(e.target.value)[0]?.value || '', columns: [] })}
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
                    <InputLabel>Columns (all if empty)</InputLabel>
                    <Select
                      multiple
                      value={op.columns}
                      onChange={(e) => updateOp(i, { columns: typeof e.target.value === 'string' ? e.target.value.split(',') : e.target.value })}
                      input={<OutlinedInput label="Columns (all if empty)" />}
                      renderValue={(selected) => (selected as string[]).length > 0 ? (selected as string[]).join(', ') : 'All columns'}
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
                <Grid item xs={12} sm={4} sx={{ display: 'flex', alignItems: 'center' }}>
                  <FormControlLabel
                    control={<Switch checked={op.remove} onChange={(e) => updateOp(i, { remove: e.target.checked })} color="error" />}
                    label={<Typography variant="body2">Remove outliers</Typography>}
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
          disabled={loading || rerunLoading}
          sx={{ borderRadius: 2, textTransform: 'none', bgcolor: '#6366f1', '&:hover': { bgcolor: '#4f46e5' } }}
        >
          {loading ? 'Running…' : 'Run Preprocessing'}
        </Button>
      </Box>

      {(loading || rerunLoading) && <LinearProgress sx={{ borderRadius: 1, mb: 2 }} />}
      {error && <Alert severity="error" sx={{ borderRadius: 2, mb: 2 }} onClose={() => setError(null)}>{error}</Alert>}

      {/* ── Persistent Operations Log ── */}
      {log.length > 0 && (
        <Box>
          <Divider sx={{ mb: 2 }} />
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="subtitle2" fontWeight={700} sx={{ color: '#374151' }}>
              Operations Log
            </Typography>
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              <Paper elevation={0} sx={{ border: '1px solid #bbf7d0', bgcolor: '#f0fdf4', borderRadius: 2, px: 2, py: 1, display: 'flex', alignItems: 'center', gap: 1 }}>
                <DoneIcon sx={{ color: '#22c55e', fontSize: 16 }} />
                <Typography variant="body2" fontWeight={700} color="#15803d">
                  {appliedCount} applied
                </Typography>
              </Paper>
              {log.filter((e) => !e.applied).length > 0 && (
                <Paper elevation={0} sx={{ border: '1px solid #e5e7eb', bgcolor: '#f9fafb', borderRadius: 2, px: 2, py: 1, display: 'flex', alignItems: 'center', gap: 1 }}>
                  <UndoIcon sx={{ color: '#9ca3af', fontSize: 16 }} />
                  <Typography variant="body2" fontWeight={700} color="#6b7280">
                    {log.filter((e) => !e.applied).length} undone
                  </Typography>
                </Paper>
              )}
            </Box>
          </Box>

          {log.map((entry, idx) => (
            <OperationResultCard
              key={entry.id}
              entry={entry}
              index={idx}
              onUndo={() => handleUndo(entry.id)}
              onRedo={() => handleRedo(entry.id)}
              onDelete={() => handleDelete(entry.id)}
              undoDisabled={rerunLoading || loading}
              redoDisabled={rerunLoading || loading}
            />
          ))}

          {appliedCount > 0 && (
            <Alert severity="info" sx={{ borderRadius: 2, mt: 1 }}>
              Processed data saved. Go to <strong>Data Preview → Processed Data</strong> tab to inspect it, or use the <strong>Compare</strong> tab to see changes.
            </Alert>
          )}
        </Box>
      )}
    </Box>
  );
}
