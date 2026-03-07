import React, { useEffect, useState } from 'react';
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
} from '@mui/material';
import {
  Storage as RowIcon,
  ViewColumn as ColIcon,
  SdStorage as SizeIcon,
  Schedule as TimeIcon,
} from '@mui/icons-material';
import { getPreview } from '../services/api';
import type { PreviewResponse } from '../types';

interface DataPreviewProps {
  sessionId: string;
}

export default function DataPreview({ sessionId }: DataPreviewProps) {
  const [data, setData] = useState<PreviewResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    getPreview(sessionId)
      .then((res) => { if (!cancelled) setData(res); })
      .catch((err) => { if (!cancelled) setError(err?.response?.data?.error || err.message); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [sessionId]);

  if (loading) {
    return (
      <Box>
        <Skeleton variant="rectangular" height={60} sx={{ borderRadius: 2, mb: 2 }} />
        <Skeleton variant="rectangular" height={300} sx={{ borderRadius: 2 }} />
      </Box>
    );
  }

  if (error) return <Alert severity="error" sx={{ borderRadius: 2 }}>{error}</Alert>;
  if (!data) return null;

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1048576).toFixed(1)} MB`;
  };

  return (
    <Box>
      {/* Metadata cards */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        {[
          { icon: <RowIcon />, value: data.metadata.total_rows.toLocaleString(), label: 'Total Rows' },
          { icon: <ColIcon />, value: data.metadata.total_columns, label: 'Columns' },
          { icon: <SizeIcon />, value: formatSize(data.metadata.file_size), label: 'File Size' },
          { icon: <TimeIcon />, value: new Date(data.metadata.upload_timestamp).toLocaleString(), label: 'Uploaded' },
        ].map((stat) => (
          <Grid item xs={6} md={3} key={stat.label}>
            <Card elevation={0} sx={{ border: '1px solid #e5e7eb', borderRadius: 2 }}>
              <CardContent sx={{ display: 'flex', alignItems: 'center', gap: 2, py: 1.5 }}>
                <Box sx={{ color: '#6366f1' }}>{stat.icon}</Box>
                <Box>
                  <Typography variant="h6" fontWeight={700} sx={{ lineHeight: 1.2 }}>
                    {stat.value}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">{stat.label}</Typography>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Preview table */}
      <Paper elevation={0} sx={{ border: '1px solid #e5e7eb', borderRadius: 2 }}>
        <Box sx={{ p: 2, borderBottom: '1px solid #e5e7eb', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="subtitle1" fontWeight={600}>Data Preview</Typography>
          <Chip
            label={`Showing ${data.preview.total_rows_shown} of ${data.preview.total_rows_available} rows`}
            size="small"
            sx={{ bgcolor: '#eef2ff', color: '#4338ca' }}
          />
        </Box>
        <TableContainer sx={{ maxHeight: 480 }}>
          <Table stickyHeader size="small">
            <TableHead>
              <TableRow>
                {data.preview.columns.map((col) => (
                  <TableCell
                    key={col}
                    sx={{ fontWeight: 700, bgcolor: '#f9fafb', whiteSpace: 'nowrap' }}
                  >
                    {col}
                  </TableCell>
                ))}
              </TableRow>
            </TableHead>
            <TableBody>
              {data.preview.rows.map((row, i) => (
                <TableRow key={i} hover>
                  {row.map((cell, j) => (
                    <TableCell key={j} sx={{ whiteSpace: 'nowrap' }}>
                      {cell === null || cell === undefined ? (
                        <Chip label="null" size="small" color="default" sx={{ height: 20, fontSize: 11 }} />
                      ) : (
                        String(cell)
                      )}
                    </TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Paper>

      {/* Column statistics */}
      {data.statistics.column_stats && Object.keys(data.statistics.column_stats).length > 0 && (
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
                {Object.entries(data.statistics.column_stats).map(([col, stats]: [string, any]) => (
                  <TableRow key={col} hover>
                    <TableCell sx={{ fontWeight: 500 }}>{col}</TableCell>
                    {['count', 'mean', 'std', 'min', 'max'].map((s) => (
                      <TableCell key={s}>
                        {stats[s] !== undefined ? (typeof stats[s] === 'number' ? stats[s].toFixed(2) : String(stats[s])) : '—'}
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
