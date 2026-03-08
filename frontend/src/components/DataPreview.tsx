import React, { useEffect, useState, useCallback } from 'react';
import {
  Box,
  Paper,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Skeleton,
  Alert,
  Card,
  CardContent,
  Grid,
  Chip,
  Tabs,
  Tab,
  Button,
  ButtonGroup,
  IconButton,
  Tooltip,
  Divider,
  CircularProgress,
} from '@mui/material';
import {
  Storage as RowIcon,
  ViewColumn as ColIcon,
  SdStorage as SizeIcon,
  Schedule as TimeIcon,
  Download as DownloadIcon,
  ChevronLeft,
  ChevronRight,
  FirstPage,
  LastPage,
  CompareArrows as CompareIcon,
} from '@mui/icons-material';
import { getPreview, getDownloadUrl } from '../services/api';
import type { PreviewResponse } from '../types';

const PAGE_SIZE = 15;

// ── Reusable paginated data table ──────────────────────────────────────────────
interface DataTableProps {
  sessionId: string;
  datasetType: 'original' | 'processed';
  label: string;
}

function DataTable({ sessionId, datasetType, label }: DataTableProps) {
  const [data, setData] = useState<PreviewResponse | null>(null);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [downloading, setDownloading] = useState<'csv' | 'json' | null>(null);

  const load = useCallback(
    (p: number) => {
      let cancelled = false;
      setLoading(true);
      getPreview(sessionId, p, PAGE_SIZE, datasetType)
        .then((res) => { if (!cancelled) { setData(res); setPage(p); } })
        .catch((err) => { if (!cancelled) setError(err?.response?.data?.error || err.message); })
        .finally(() => { if (!cancelled) setLoading(false); });
      return () => { cancelled = true; };
    },
    [sessionId, datasetType]
  );

  useEffect(() => { load(1); }, [load]);

  const handleDownload = async (format: 'csv' | 'json') => {
    setDownloading(format);
    try {
      if (format === 'csv') {
        const url = await getDownloadUrl(sessionId, datasetType, 'csv');
        window.open(url, '_blank');
      } else {
        // JSON: trigger API → server converts and returns JSON body
        const url = await getDownloadUrl(sessionId, datasetType, 'json');
        window.open(url, '_blank');
      }
    } catch (err: any) {
      alert(`Download failed: ${err?.response?.data?.error || err.message}`);
    } finally {
      setDownloading(null);
    }
  };

  if (loading && !data) {
    return (
      <Box>
        <Skeleton variant="rectangular" height={40} sx={{ borderRadius: 1, mb: 1 }} />
        <Skeleton variant="rectangular" height={260} sx={{ borderRadius: 1 }} />
      </Box>
    );
  }

  if (error) return <Alert severity="error" sx={{ borderRadius: 2 }}>{error}</Alert>;
  if (!data) return null;

  const { preview } = data;
  const totalPages = preview.total_pages ?? 1;
  const currentPage = preview.current_page ?? page;

  return (
    <Paper elevation={0} sx={{ border: '1px solid #e5e7eb', borderRadius: 2 }}>
      {/* Header row */}
      <Box sx={{ px: 2, py: 1.5, borderBottom: '1px solid #e5e7eb', display: 'flex', flexWrap: 'wrap', gap: 1, alignItems: 'center', justifyContent: 'space-between' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Typography variant="subtitle2" fontWeight={700}>{label}</Typography>
          <Chip
            label={`${preview.total_rows_available.toLocaleString()} rows × ${preview.columns.length} cols`}
            size="small"
            sx={{ bgcolor: '#eef2ff', color: '#4338ca' }}
          />
        </Box>
        {/* Download buttons */}
        <ButtonGroup size="small" variant="outlined" disabled={downloading !== null}>
          <Tooltip title="Download as CSV">
            <Button
              startIcon={downloading === 'csv' ? <CircularProgress size={14} /> : <DownloadIcon />}
              onClick={() => handleDownload('csv')}
              sx={{ textTransform: 'none', fontSize: 12 }}
            >
              CSV
            </Button>
          </Tooltip>
          <Tooltip title="Download as JSON">
            <Button
              startIcon={downloading === 'json' ? <CircularProgress size={14} /> : <DownloadIcon />}
              onClick={() => handleDownload('json')}
              sx={{ textTransform: 'none', fontSize: 12 }}
            >
              JSON
            </Button>
          </Tooltip>
        </ButtonGroup>
      </Box>

      {/* Table */}
      <TableContainer sx={{ maxHeight: 440, position: 'relative' }}>
        {loading && (
          <Box sx={{ position: 'absolute', inset: 0, bgcolor: 'rgba(255,255,255,0.6)', zIndex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <CircularProgress size={28} />
          </Box>
        )}
        <Table stickyHeader size="small">
          <TableHead>
            <TableRow>
              <TableCell sx={{ fontWeight: 700, bgcolor: '#f9fafb', color: '#6b7280', width: 50, textAlign: 'center' }}>#</TableCell>
              {preview.columns.map((col) => (
                <TableCell key={col} sx={{ fontWeight: 700, bgcolor: '#f9fafb', whiteSpace: 'nowrap' }}>
                  {col}
                </TableCell>
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            {preview.rows.map((row, i) => {
              const globalRowNum = (currentPage - 1) * PAGE_SIZE + i + 1;
              return (
                <TableRow key={i} hover>
                  <TableCell sx={{ color: '#9ca3af', textAlign: 'center', fontSize: 11 }}>{globalRowNum}</TableCell>
                  {row.map((cell, j) => (
                    <TableCell key={j} sx={{ whiteSpace: 'nowrap', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                      {cell === null || cell === undefined ? (
                        <Chip label="null" size="small" sx={{ height: 18, fontSize: 10, bgcolor: '#fef3c7', color: '#92400e' }} />
                      ) : (
                        String(cell)
                      )}
                    </TableCell>
                  ))}
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </TableContainer>

      {/* Pagination controls */}
      <Box sx={{ px: 2, py: 1, borderTop: '1px solid #e5e7eb', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 1 }}>
        <Typography variant="caption" color="text.secondary">
          Page {currentPage} of {totalPages} &nbsp;·&nbsp; rows {((currentPage - 1) * PAGE_SIZE) + 1}–{Math.min(currentPage * PAGE_SIZE, preview.total_rows_available)} of {preview.total_rows_available.toLocaleString()}
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
          <IconButton size="small" onClick={() => load(1)} disabled={currentPage <= 1 || loading}>
            <FirstPage fontSize="small" />
          </IconButton>
          <IconButton size="small" onClick={() => load(currentPage - 1)} disabled={currentPage <= 1 || loading}>
            <ChevronLeft fontSize="small" />
          </IconButton>
          {/* Page number chips */}
          {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
            let p: number;
            if (totalPages <= 5) p = i + 1;
            else if (currentPage <= 3) p = i + 1;
            else if (currentPage >= totalPages - 2) p = totalPages - 4 + i;
            else p = currentPage - 2 + i;
            return (
              <Chip
                key={p}
                label={p}
                size="small"
                onClick={() => load(p)}
                sx={{
                  cursor: 'pointer',
                  bgcolor: p === currentPage ? '#6366f1' : '#f3f4f6',
                  color: p === currentPage ? '#fff' : '#374151',
                  fontWeight: p === currentPage ? 700 : 400,
                  minWidth: 32,
                  height: 28,
                  '&:hover': { bgcolor: p === currentPage ? '#4f46e5' : '#e5e7eb' },
                }}
              />
            );
          })}
          <IconButton size="small" onClick={() => load(currentPage + 1)} disabled={currentPage >= totalPages || loading}>
            <ChevronRight fontSize="small" />
          </IconButton>
          <IconButton size="small" onClick={() => load(totalPages)} disabled={currentPage >= totalPages || loading}>
            <LastPage fontSize="small" />
          </IconButton>
        </Box>
      </Box>
    </Paper>
  );
}

// ── Compare summary panel ──────────────────────────────────────────────────────
interface CompareProps {
  sessionId: string;
}

function CompareView({ sessionId }: CompareProps) {
  const [orig, setOrig] = useState<PreviewResponse | null>(null);
  const [proc, setProc] = useState<PreviewResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    Promise.all([
      getPreview(sessionId, 1, PAGE_SIZE, 'original'),
      getPreview(sessionId, 1, PAGE_SIZE, 'processed'),
    ])
      .then(([o, p]) => { if (!cancelled) { setOrig(o); setProc(p); } })
      .catch((err) => { if (!cancelled) setError(err?.response?.data?.error || err.message); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [sessionId]);

  if (loading) return <Skeleton variant="rectangular" height={200} sx={{ borderRadius: 2 }} />;
  if (error) return <Alert severity="error">{error}</Alert>;
  if (!orig || !proc) return null;

  const metrics = [
    { label: 'Total Rows', orig: orig.metadata.total_rows, proc: proc.metadata.total_rows },
    { label: 'Total Columns', orig: orig.metadata.total_columns, proc: proc.metadata.total_columns },
    {
      label: 'Missing Values',
      orig: (orig.statistics as any)?.missing_value_summary?.total_missing_values ?? '—',
      proc: (proc.statistics as any)?.missing_value_summary?.total_missing_values ?? '—',
    },
    {
      label: 'Missing %',
      orig: `${(orig.statistics as any)?.missing_value_summary?.missing_percentage ?? 0}%`,
      proc: `${(proc.statistics as any)?.missing_value_summary?.missing_percentage ?? 0}%`,
    },
  ];

  return (
    <Box>
      {/* Summary comparison */}
      <Paper elevation={0} sx={{ border: '1px solid #e5e7eb', borderRadius: 2, mb: 3 }}>
        <Box sx={{ p: 2, borderBottom: '1px solid #e5e7eb', display: 'flex', alignItems: 'center', gap: 1 }}>
          <CompareIcon sx={{ color: '#6366f1' }} />
          <Typography variant="subtitle1" fontWeight={700}>Dataset Comparison</Typography>
        </Box>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell sx={{ fontWeight: 700, bgcolor: '#f9fafb', width: '34%' }}>Metric</TableCell>
              <TableCell sx={{ fontWeight: 700, bgcolor: '#f0fdf4', color: '#15803d' }}>Original</TableCell>
              <TableCell sx={{ fontWeight: 700, bgcolor: '#eff6ff', color: '#1d4ed8' }}>Processed</TableCell>
              <TableCell sx={{ fontWeight: 700, bgcolor: '#f9fafb' }}>Change</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {metrics.map((m) => {
              const origNum = typeof m.orig === 'number' ? m.orig : parseFloat(String(m.orig));
              const procNum = typeof m.proc === 'number' ? m.proc : parseFloat(String(m.proc));
              const diff = !isNaN(origNum) && !isNaN(procNum) ? procNum - origNum : null;
              return (
                <TableRow key={m.label} hover>
                  <TableCell sx={{ fontWeight: 500 }}>{m.label}</TableCell>
                  <TableCell sx={{ bgcolor: '#f0fdf4' }}>{typeof m.orig === 'number' ? m.orig.toLocaleString() : m.orig}</TableCell>
                  <TableCell sx={{ bgcolor: '#eff6ff' }}>{typeof m.proc === 'number' ? m.proc.toLocaleString() : m.proc}</TableCell>
                  <TableCell>
                    {diff !== null ? (
                      <Chip
                        label={`${diff >= 0 ? '+' : ''}${typeof m.orig === 'string' && m.orig.endsWith('%') ? diff.toFixed(2) + '%' : diff.toLocaleString()}`}
                        size="small"
                        sx={{ bgcolor: diff < 0 ? '#dcfce7' : diff > 0 ? '#fef2f2' : '#f3f4f6', color: diff < 0 ? '#15803d' : diff > 0 ? '#b91c1c' : '#6b7280', height: 20, fontSize: 11, fontWeight: 600 }}
                      />
                    ) : '—'}
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </Paper>

      {/* Side-by-side first rows */}
      <Grid container spacing={2}>
        <Grid item xs={12} md={6}>
          <Typography variant="caption" fontWeight={700} sx={{ color: '#15803d', display: 'block', mb: 0.5 }}>ORIGINAL (first {PAGE_SIZE} rows)</Typography>
          <DataTable sessionId={sessionId} datasetType="original" label="Original Data" />
        </Grid>
        <Grid item xs={12} md={6}>
          <Typography variant="caption" fontWeight={700} sx={{ color: '#1d4ed8', display: 'block', mb: 0.5 }}>PROCESSED (first {PAGE_SIZE} rows)</Typography>
          <DataTable sessionId={sessionId} datasetType="processed" label="Processed Data" />
        </Grid>
      </Grid>
    </Box>
  );
}

// ── Main DataPreview component ─────────────────────────────────────────────────
interface DataPreviewProps {
  sessionId: string;
  hasProcessed?: boolean; // lifted from App — updates immediately after preprocessing
}

export default function DataPreview({ sessionId, hasProcessed: hasProcessedProp }: DataPreviewProps) {
  const [meta, setMeta] = useState<PreviewResponse | null>(null);
  const [metaLoading, setMetaLoading] = useState(true);
  const [metaError, setMetaError] = useState<string | null>(null);
  const [tab, setTab] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setMetaLoading(true);
    getPreview(sessionId, 1, PAGE_SIZE, 'original')
      .then((res) => { if (!cancelled) setMeta(res); })
      .catch((err) => { if (!cancelled) setMetaError(err?.response?.data?.error || err.message); })
      .finally(() => { if (!cancelled) setMetaLoading(false); });
    return () => { cancelled = true; };
  }, [sessionId]);

  if (metaLoading) {
    return (
      <Box>
        <Skeleton variant="rectangular" height={80} sx={{ borderRadius: 2, mb: 2 }} />
        <Skeleton variant="rectangular" height={350} sx={{ borderRadius: 2 }} />
      </Box>
    );
  }

  if (metaError) return <Alert severity="error" sx={{ borderRadius: 2 }}>{metaError}</Alert>;
  if (!meta) return null;

  // Use prop if provided (set by App after preprocessing); fall back to API metadata
  const hasProcessed = hasProcessedProp ?? (meta.metadata.has_processed_data === true);

  const formatSize = (bytes: number) => {
    if (!bytes) return '—';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1048576).toFixed(1)} MB`;
  };

  return (
    <Box>
      {/* Metadata cards */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        {[
          { icon: <RowIcon />, value: meta.metadata.total_rows.toLocaleString(), label: 'Total Rows' },
          { icon: <ColIcon />, value: meta.metadata.total_columns, label: 'Columns' },
          { icon: <SizeIcon />, value: formatSize(meta.metadata.file_size), label: 'File Size' },
          { icon: <TimeIcon />, value: new Date(meta.metadata.upload_timestamp?.endsWith('Z') || meta.metadata.upload_timestamp?.includes('+') ? meta.metadata.upload_timestamp : (meta.metadata.upload_timestamp ?? '') + 'Z').toLocaleString(), label: 'Uploaded' },
        ].map((stat) => (
          <Grid item xs={6} md={3} key={stat.label}>
            <Card elevation={0} sx={{ border: '1px solid #e5e7eb', borderRadius: 2 }}>
              <CardContent sx={{ display: 'flex', alignItems: 'center', gap: 2, py: 1.5, '&:last-child': { pb: 1.5 } }}>
                <Box sx={{ color: '#6366f1' }}>{stat.icon}</Box>
                <Box>
                  <Typography variant="h6" fontWeight={700} sx={{ lineHeight: 1.2 }}>{stat.value}</Typography>
                  <Typography variant="caption" color="text.secondary">{stat.label}</Typography>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Tabs */}
      <Paper elevation={0} sx={{ border: '1px solid #e5e7eb', borderRadius: 2, overflow: 'hidden' }}>
        <Tabs
          value={tab}
          onChange={(_, v) => setTab(v)}
          sx={{
            borderBottom: '1px solid #e5e7eb',
            minHeight: 44,
            '& .MuiTab-root': { textTransform: 'none', minHeight: 44, fontWeight: 600, fontSize: 13 },
            '& .Mui-selected': { color: '#6366f1' },
            '& .MuiTabs-indicator': { bgcolor: '#6366f1' },
          }}
        >
          <Tab label="Original Data" />
          <Tab label="Processed Data" disabled={!hasProcessed} />
          <Tab label="Compare" disabled={!hasProcessed} icon={<CompareIcon sx={{ fontSize: 16 }} />} iconPosition="end" />
        </Tabs>

        <Box sx={{ p: 2 }}>
          {tab === 0 && <DataTable sessionId={sessionId} datasetType="original" label="Original Data" />}
          {tab === 1 && hasProcessed && <DataTable sessionId={sessionId} datasetType="processed" label="Processed Data" />}
          {tab === 1 && !hasProcessed && (
            <Alert severity="info" sx={{ borderRadius: 2 }}>
              No processed data yet. Run preprocessing operations first to generate processed data.
            </Alert>
          )}
          {tab === 2 && hasProcessed && <CompareView sessionId={sessionId} />}
          {tab === 2 && !hasProcessed && (
            <Alert severity="info" sx={{ borderRadius: 2 }}>
              No processed data yet. Run preprocessing operations to enable comparison.
            </Alert>
          )}
        </Box>
      </Paper>

      {/* Column statistics (full-width, shown for original always) */}
      {meta.statistics.column_stats && Object.keys(meta.statistics.column_stats).length > 0 && (
        <Paper elevation={0} sx={{ mt: 3, border: '1px solid #e5e7eb', borderRadius: 2 }}>
          <Box sx={{ p: 2, borderBottom: '1px solid #e5e7eb' }}>
            <Typography variant="subtitle1" fontWeight={600}>Column Statistics</Typography>
          </Box>
          <TableContainer>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell sx={{ fontWeight: 700, bgcolor: '#f9fafb' }}>Column</TableCell>
                  {['count', 'mean', 'std', 'min', 'max'].map((s) => (
                    <TableCell key={s} sx={{ fontWeight: 700, bgcolor: '#f9fafb' }}>{s}</TableCell>
                  ))}
                </TableRow>
              </TableHead>
              <TableBody>
                {Object.entries(meta.statistics.column_stats).map(([col, stats]: [string, any]) => (
                  <TableRow key={col} hover>
                    <TableCell sx={{ fontWeight: 500 }}>{col}</TableCell>
                    {['count', 'mean', 'std', 'min', 'max'].map((s) => (
                      <TableCell key={s}>
                        {stats[s] !== undefined
                          ? typeof stats[s] === 'number' ? stats[s].toFixed(2) : String(stats[s])
                          : '—'}
                      </TableCell>
                    ))}
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
