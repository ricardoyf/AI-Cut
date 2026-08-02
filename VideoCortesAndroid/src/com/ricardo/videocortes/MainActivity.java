package com.ricardo.videocortes;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Intent;
import android.content.SharedPreferences;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.media.MediaExtractor;
import android.media.MediaFormat;
import android.media.MediaMetadataRetriever;
import android.media.MediaMuxer;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.ParcelFileDescriptor;
import android.provider.DocumentsContract;
import android.util.Size;
import android.view.Gravity;
import android.view.KeyEvent;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.HorizontalScrollView;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.SeekBar;
import android.widget.ScrollView;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.text.DateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Locale;
import java.util.Set;

import androidx.media3.common.MediaItem;
import androidx.media3.exoplayer.ExoPlayer;
import androidx.media3.ui.AspectRatioFrameLayout;
import androidx.media3.ui.PlayerView;

public class MainActivity extends Activity {
    private static final int REQ_TREE = 31;
    private static final String PREFS = "videocortes_estado";
    private static final String BAD_SUFFIX = "_MALO.txt";
    private static final String LLC_MARK = "-LLC-";

    private final ArrayList<VideoItem> videos = new ArrayList<>();
    private final ArrayList<Segment> segments = new ArrayList<>();
    private final Handler handler = new Handler(Looper.getMainLooper());

    private TextView info;
    private static final String APP_VERSION = "v1.6";

    private PlayerView playerView;
    private SegmentBar segmentBar;
    private SeekBar positionSeek;
    private TextView segmentText;
    private Switch badSwitch;
    private ExoPlayer player;
    private Uri treeUri;
    private int index = 0;
    private long cutStartMs = 0;
    private long cutEndMs = -1;
    private long durationMs = 0;
    private boolean loadingSwitch = false;
    private boolean exporting = false;
    private boolean userScrubbing = false;
    private int openGeneration = 0;

    private final Runnable ticker = new Runnable() {
        @Override
        public void run() {
            updateInfo();
            updateSegmentsUi();
            handler.postDelayed(this, 250);
        }
    };

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        buildUi();
        Uri saved = getSavedTree();
        if (saved != null) {
            try {
                loadFolder(saved);
            } catch (Exception e) {
                info.setText("Elige carpeta. No he podido reabrir la carpeta anterior.");
            }
        } else {
            info.setText("Elige carpeta. Se mostraran primero los videos mas recientes.");
        }
        handler.post(ticker);
    }

    @Override
    protected void onDestroy() {
        handler.removeCallbacksAndMessages(null);
        releasePlayer();
        super.onDestroy();
    }

    private void buildUi() {
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(0xff050505);

        LinearLayout header = new LinearLayout(this);
        header.setOrientation(LinearLayout.HORIZONTAL);
        header.setGravity(Gravity.CENTER_VERTICAL);
        header.setPadding(0, 0, dp(8), 0);

        info = new TextView(this);
        info.setTextColor(Color.WHITE);
        info.setTextSize(14);
        info.setPadding(dp(12), dp(8), dp(12), dp(6));
        header.addView(info, new LinearLayout.LayoutParams(
                0,
                LinearLayout.LayoutParams.WRAP_CONTENT,
                1));

        TextView version = new TextView(this);
        version.setText(APP_VERSION);
        version.setTextColor(0xff9ca3af);
        version.setTextSize(11);
        version.setGravity(Gravity.RIGHT);
        header.addView(version, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT));
        root.addView(header, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT));

        playerView = new PlayerView(this);
        playerView.setBackgroundColor(Color.BLACK);
        playerView.setUseController(false);
        playerView.setResizeMode(AspectRatioFrameLayout.RESIZE_MODE_FIT);
        playerView.setKeepScreenOn(true);
        playerView.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View v) { togglePlay(); }
        });
        root.addView(playerView, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 0, 1));

        segmentBar = new SegmentBar(this);
        segmentBar.setOnTouchListener(new View.OnTouchListener() {
            @Override public boolean onTouch(View v, MotionEvent event) {
                if (event.getAction() == MotionEvent.ACTION_DOWN
                        || event.getAction() == MotionEvent.ACTION_MOVE
                        || event.getAction() == MotionEvent.ACTION_UP) {
                    seekToFraction(event.getX() / Math.max(1f, v.getWidth()));
                    return true;
                }
                return false;
            }
        });
        root.addView(segmentBar, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                dp(32)));

        positionSeek = new SeekBar(this);
        positionSeek.setMax(10000);
        positionSeek.setPadding(dp(8), 0, dp(8), 0);
        positionSeek.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                if (fromUser) seekToFraction(progress / 10000f);
            }
            @Override public void onStartTrackingTouch(SeekBar seekBar) {
                userScrubbing = true;
            }
            @Override public void onStopTrackingTouch(SeekBar seekBar) {
                userScrubbing = false;
                seekToFraction(seekBar.getProgress() / 10000f);
            }
        });
        root.addView(positionSeek, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                dp(38)));

        HorizontalScrollView segmentScroll = new HorizontalScrollView(this);
        segmentScroll.setBackgroundColor(0xff0b0b0b);
        segmentText = new TextView(this);
        segmentText.setTextColor(0xffe5e7eb);
        segmentText.setTextSize(13);
        segmentText.setPadding(dp(10), dp(5), dp(10), dp(5));
        segmentScroll.addView(segmentText);
        root.addView(segmentScroll, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT));

        LinearLayout controls = new LinearLayout(this);
        controls.setOrientation(LinearLayout.VERTICAL);
        controls.setPadding(dp(6), dp(6), dp(6), dp(10));
        controls.setBackgroundColor(0xff111111);

        LinearLayout top = row();
        Button folder = button("Carpeta");
        Button choose = button("Videos");
        Button prev = button("Ant");
        Button next = button("N");
        badSwitch = new Switch(this);
        badSwitch.setText("Malo");
        badSwitch.setTextColor(Color.WHITE);
        badSwitch.setTextSize(14);
        badSwitch.setPadding(dp(8), 0, dp(8), 0);
        top.addView(folder);
        top.addView(choose);
        top.addView(prev);
        top.addView(next);
        top.addView(badSwitch);

        LinearLayout keys = row();
        Button play = button("P");
        Button markIn = button("I");
        Button markOut = button("O");
        Button export = button("E");
        Button reset = button("Reset");
        keys.addView(play);
        keys.addView(markIn);
        keys.addView(markOut);
        keys.addView(export);
        keys.addView(reset);

        controls.addView(top);
        controls.addView(keys);
        root.addView(controls);
        setContentView(root);

        folder.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View v) { pickFolder(); }
        });
        choose.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View v) { chooseVideo(); }
        });
        prev.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View v) { previousVideo(); }
        });
        next.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View v) { nextVideo(); }
        });
        play.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View v) { togglePlay(); }
        });
        markIn.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View v) { markIn(); }
        });
        markOut.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View v) { markOut(); }
        });
        export.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View v) { exportCut(); }
        });
        reset.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View v) { resetMarks(); }
        });
        badSwitch.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                if (loadingSwitch || !isChecked) return;
                markBadAndNext();
            }
        });
    }

    private LinearLayout row() {
        LinearLayout row = new LinearLayout(this);
        row.setGravity(Gravity.CENTER);
        row.setOrientation(LinearLayout.HORIZONTAL);
        return row;
    }

    private Button button(String text) {
        Button b = new Button(this);
        b.setText(text);
        b.setTextSize(13);
        b.setAllCaps(false);
        b.setTextColor(Color.WHITE);
        b.setBackgroundColor(0xff1f2937);
        b.setPadding(dp(6), 0, dp(6), 0);
        b.setMinHeight(dp(44));
        b.setMinWidth(dp(44));
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(0, dp(48), 1);
        lp.setMargins(dp(3), dp(3), dp(3), dp(3));
        b.setLayoutParams(lp);
        return b;
    }

    private int dp(int v) {
        return (int) (v * getResources().getDisplayMetrics().density + 0.5f);
    }

    private void pickFolder() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT_TREE);
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION
                | Intent.FLAG_GRANT_WRITE_URI_PERMISSION
                | Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION
                | Intent.FLAG_GRANT_PREFIX_URI_PERMISSION);
        startActivityForResult(intent, REQ_TREE);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQ_TREE && resultCode == RESULT_OK && data != null) {
            Uri uri = data.getData();
            if (uri == null) return;
            int flags = data.getFlags() & (Intent.FLAG_GRANT_READ_URI_PERMISSION
                    | Intent.FLAG_GRANT_WRITE_URI_PERMISSION);
            getContentResolver().takePersistableUriPermission(uri, flags);
            getPreferences().edit().putString("tree", uri.toString()).apply();
            loadFolder(uri);
        }
    }

    private SharedPreferences getPreferences() {
        return getSharedPreferences(PREFS, MODE_PRIVATE);
    }

    private Uri getSavedTree() {
        String raw = getPreferences().getString("tree", null);
        return raw == null ? null : Uri.parse(raw);
    }

    private void loadFolder(final Uri uri) {
        treeUri = uri;
        info.setText("Escaneando videos...");
        releasePlayer();
        videos.clear();
        segments.clear();
        index = 0;
        updateSegmentsUi();
        new Thread(new Runnable() {
            @Override public void run() {
                final ArrayList<VideoItem> found = new ArrayList<>();
                try {
                    String rootId = DocumentsContract.getTreeDocumentId(uri);
                    scanFolder(rootId, found, 0);
                    sortVideosNewestFirst(found);
                } catch (Exception e) {
                    showToast("Error al escanear: " + e.getMessage());
                }
                runOnUiThread(new Runnable() {
                    @Override public void run() {
                        videos.clear();
                        videos.addAll(found);
                        if (videos.isEmpty()) {
                            info.setText("No quedan videos pendientes. Se omiten los que ya tienen -LLC- o MALO.");
                            segmentText.setText("");
                        } else {
                            openCurrent(true);
                        }
                    }
                });
            }
        }, "videocortes-scan").start();
    }

    private void sortVideosNewestFirst(ArrayList<VideoItem> target) {
        Collections.sort(target, new Comparator<VideoItem>() {
            @Override public int compare(VideoItem a, VideoItem b) {
                int byDate = Long.compare(b.modified, a.modified);
                if (byDate != 0) return byDate;
                return a.name.compareToIgnoreCase(b.name);
            }
        });
    }

    private void scanFolder(String parentDocId, ArrayList<VideoItem> out, int depth) {
        if (depth > 6 || treeUri == null) return;
        Uri childrenUri = DocumentsContract.buildChildDocumentsUriUsingTree(treeUri, parentDocId);
        Cursor c = null;
        ArrayList<Doc> docs = new ArrayList<>();
        Set<String> names = new HashSet<>();
        try {
            c = getContentResolver().query(childrenUri, new String[] {
                    DocumentsContract.Document.COLUMN_DOCUMENT_ID,
                    DocumentsContract.Document.COLUMN_DISPLAY_NAME,
                    DocumentsContract.Document.COLUMN_MIME_TYPE,
                    DocumentsContract.Document.COLUMN_LAST_MODIFIED
            }, null, null, DocumentsContract.Document.COLUMN_DISPLAY_NAME + " ASC");
            if (c == null) return;
            while (c.moveToNext()) {
                Doc d = new Doc();
                d.id = c.getString(0);
                d.name = c.getString(1);
                d.mime = c.getString(2);
                d.modified = c.isNull(3) ? 0L : c.getLong(3);
                if (d.name == null || d.id == null) continue;
                docs.add(d);
                names.add(d.name);
            }
        } finally {
            if (c != null) c.close();
        }

        for (Doc d : docs) {
            if (DocumentsContract.Document.MIME_TYPE_DIR.equals(d.mime)) {
                scanFolder(d.id, out, depth + 1);
            }
        }
        for (Doc d : docs) {
            if (!isVideo(d)) continue;
            String base = baseName(d.name);
            if (names.contains(base + BAD_SUFFIX)) continue;
            if (hasLlcOutput(names, base)) continue;
            VideoItem item = new VideoItem();
            item.name = d.name;
            item.base = base;
            item.uri = DocumentsContract.buildDocumentUriUsingTree(treeUri, d.id);
            item.parentDocId = parentDocId;
            item.modified = d.modified;
            out.add(item);
        }
    }

    private boolean isVideo(Doc d) {
        if (d.name == null) return false;
        if (d.name.contains(LLC_MARK)) return false;
        if (d.mime != null && d.mime.startsWith("video/")) return true;
        String n = d.name.toLowerCase(Locale.US);
        return n.endsWith(".mp4") || n.endsWith(".mov") || n.endsWith(".mkv")
                || n.endsWith(".3gp") || n.endsWith(".webm") || n.endsWith(".m4v");
    }

    private boolean hasLlcOutput(Set<String> names, String base) {
        String prefix = base + LLC_MARK;
        for (String name : names) {
            if (name.startsWith(prefix)) return true;
        }
        return false;
    }

    private void openCurrent(boolean resetSegments) {
        if (videos.isEmpty()) return;
        if (index < 0) index = 0;
        if (index >= videos.size()) index = videos.size() - 1;
        if (resetSegments) {
            cutStartMs = 0;
            cutEndMs = -1;
            segments.clear();
        }
        VideoItem item = videos.get(index);
        durationMs = readDuration(item.uri);
        loadingSwitch = true;
        badSwitch.setChecked(false);
        loadingSwitch = false;
        preparePlayer(item.uri);
        updateInfo();
        updateSegmentsUi();
    }

    private void preparePlayer(final Uri uri) {
        final int generation = ++openGeneration;
        releasePlayer();
        try {
            player = new ExoPlayer.Builder(this).build();
            playerView.setPlayer(player);
            player.setMediaItem(MediaItem.fromUri(uri));
            player.prepare();
            player.seekTo(0);
        player.setPlayWhenReady(false);
            if (generation == openGeneration) updateInfo();
        } catch (Exception e) {
            if (generation == openGeneration) showToast("Error al abrir reproductor: " + e.getMessage());
        }
    }

    private void releasePlayer() {
        if (player != null) {
            try { if (playerView != null) playerView.setPlayer(null); } catch (Exception ignored) {}
            try { player.release(); } catch (Exception ignored) {}
            player = null;
        }
    }

    private long currentPosition() {
        return player == null ? 0 : Math.max(0, player.getCurrentPosition());
    }

    private boolean isPlaying() {
        return player != null && player.isPlaying();
    }

    private void seekToFraction(float fraction) {
        if (durationMs <= 0) return;
        float clamped = Math.max(0f, Math.min(1f, fraction));
        seekToMs((long) (durationMs * clamped));
    }

    private void seekToMs(long positionMs) {
        if (player == null) return;
        long target = Math.max(0, durationMs > 0 ? Math.min(durationMs, positionMs) : positionMs);
        try {
            player.seekTo(target);
        } catch (Exception ignored) {
        }
        updateInfo();
        updateSegmentsUi();
    }

    private long readDuration(Uri uri) {
        MediaMetadataRetriever r = new MediaMetadataRetriever();
        try {
            r.setDataSource(this, uri);
            String raw = r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION);
            return raw == null ? 0 : Long.parseLong(raw);
        } catch (Exception e) {
            return 0;
        } finally {
            try { r.release(); } catch (Exception ignored) {}
        }
    }

    private void togglePlay() {
        if (videos.isEmpty() || player == null) return;
        if (player.isPlaying()) player.pause();
        else player.play();
        updateInfo();
    }

    private void markIn() {
        cutStartMs = currentPosition();
        cutEndMs = -1;
        showToast("I " + fmt(cutStartMs));
        updateInfo();
        updateSegmentsUi();
    }

    private void markOut() {
        cutEndMs = currentPosition();
        if (cutEndMs <= cutStartMs + 200) {
            showToast("O debe estar despues de I");
            return;
        }
        segments.add(new Segment(cutStartMs, cutEndMs));
        showToast("Segmento " + segments.size() + "  " + fmt(cutStartMs) + " - " + fmt(cutEndMs));
        cutStartMs = cutEndMs;
        cutEndMs = -1;
        updateInfo();
        updateSegmentsUi();
    }

    private void resetMarks() {
        cutStartMs = currentPosition();
        cutEndMs = -1;
        segments.clear();
        showToast("Segmentos borrados");
        updateInfo();
        updateSegmentsUi();
    }

    private void nextVideo() {
        if (videos.isEmpty()) return;
        if (index < videos.size() - 1) {
            index++;
            openCurrent(true);
        } else {
            showToast("Fin de la lista");
        }
    }

    private void previousVideo() {
        if (videos.isEmpty()) return;
        if (index > 0) {
            index--;
            openCurrent(true);
        }
    }

    private void chooseVideo() {
        if (videos.isEmpty()) return;
        final AlertDialog[] dialogRef = new AlertDialog[1];
        DateFormat format = DateFormat.getDateTimeInstance(
                DateFormat.SHORT,
                DateFormat.SHORT,
                Locale.getDefault());
        ScrollView scroll = new ScrollView(this);
        LinearLayout list = new LinearLayout(this);
        list.setOrientation(LinearLayout.VERTICAL);
        int rowPad = dp(8);
        for (int i = 0; i < videos.size(); i++) {
            final int videoIndex = i;
            VideoItem item = videos.get(i);
            String date = item.modified > 0 ? format.format(new Date(item.modified)) : "sin fecha";
            LinearLayout row = new LinearLayout(this);
            row.setOrientation(LinearLayout.HORIZONTAL);
            row.setGravity(Gravity.CENTER_VERTICAL);
            row.setPadding(rowPad, rowPad, rowPad, rowPad);

            final ImageView thumb = new ImageView(this);
            thumb.setScaleType(ImageView.ScaleType.CENTER_CROP);
            thumb.setBackgroundColor(0xff333333);
            row.addView(thumb, new LinearLayout.LayoutParams(dp(84), dp(64)));
            loadVideoThumbnailAsync(item.uri, thumb);

            TextView label = new TextView(this);
            label.setText((i + 1) + "/" + videos.size() + "  " + date + "\n" + item.name);
            label.setTextColor(0xff202020);
            label.setTextSize(14);
            label.setPadding(dp(10), 0, 0, 0);
            row.addView(label, new LinearLayout.LayoutParams(
                    0,
                    LinearLayout.LayoutParams.WRAP_CONTENT,
                    1));
            row.setOnClickListener(new View.OnClickListener() {
                @Override public void onClick(View v) {
                    index = videoIndex;
                    openCurrent(true);
                    if (dialogRef[0] != null) dialogRef[0].dismiss();
                }
            });
            list.addView(row, new LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT));
        }
        scroll.addView(list);
        dialogRef[0] = new AlertDialog.Builder(this)
                .setTitle("Elegir video")
                .setView(scroll)
                .create();
        dialogRef[0].show();
    }

    private void loadVideoThumbnailAsync(final Uri uri, final ImageView target) {
        new Thread(new Runnable() {
            @Override public void run() {
                final Bitmap bitmap = loadVideoThumbnail(uri);
                if (bitmap == null) return;
                runOnUiThread(new Runnable() {
                    @Override public void run() { target.setImageBitmap(bitmap); }
                });
            }
        }, "videocortes-thumb").start();
    }

    private Bitmap loadVideoThumbnail(Uri uri) {
        try {
            if (android.os.Build.VERSION.SDK_INT >= 29) {
                return getContentResolver().loadThumbnail(uri, new Size(dp(160), dp(120)), null);
            }
        } catch (Exception ignored) {
        }
        MediaMetadataRetriever r = new MediaMetadataRetriever();
        try {
            r.setDataSource(this, uri);
            return r.getFrameAtTime(1000000, MediaMetadataRetriever.OPTION_CLOSEST_SYNC);
        } catch (Exception e) {
            return null;
        } finally {
            try { r.release(); } catch (Exception ignored) {}
        }
    }

    private void markBadAndNext() {
        if (videos.isEmpty()) return;
        final VideoItem item = videos.get(index);
        new Thread(new Runnable() {
            @Override public void run() {
                try {
                    createTextFile(item.parentDocId, item.base + BAD_SUFFIX, "MALO\n" + item.name + "\n");
                    runOnUiThread(new Runnable() {
                        @Override public void run() {
                            showToast("Marcado MALO");
                            videos.remove(index);
                            if (index >= videos.size()) index = Math.max(0, videos.size() - 1);
                            if (videos.isEmpty()) {
                                releasePlayer();
                                info.setText("No quedan videos pendientes.");
                            } else {
                                openCurrent(true);
                            }
                        }
                    });
                } catch (Exception e) {
                    showToast("No pude marcar MALO: " + e.getMessage());
                }
            }
        }).start();
    }

    private void exportCut() {
        if (videos.isEmpty() || exporting) return;
        final VideoItem item = videos.get(index);
        final ArrayList<Segment> toExport = new ArrayList<>(segments);
        if (toExport.isEmpty() && cutEndMs > cutStartMs + 200) {
            toExport.add(new Segment(cutStartMs, cutEndMs));
        }
        if (toExport.isEmpty()) {
            showToast("Marca I y O antes de exportar");
            return;
        }
        exporting = true;
        info.setText("Exportando " + toExport.size() + " segmento(s)...");
        new Thread(new Runnable() {
            @Override public void run() {
                try {
                    for (Segment s : toExport) {
                        String outName = uniqueName(item.parentDocId,
                                safeName(item.base + LLC_MARK + fmt(s.startMs) + "-" + fmt(s.endMs) + ".mp4"));
                        Uri outUri = DocumentsContract.createDocument(
                                getContentResolver(),
                                DocumentsContract.buildDocumentUriUsingTree(treeUri, item.parentDocId),
                                "video/mp4",
                                outName);
                        if (outUri == null) throw new Exception("No se pudo crear salida");
                        trimVideo(item.uri, outUri, s.startMs * 1000L, s.endMs * 1000L);
                    }
                    runOnUiThread(new Runnable() {
                        @Override public void run() {
                            exporting = false;
                            showToast("Exportado LLC");
                            videos.remove(index);
                            if (index >= videos.size()) index = Math.max(0, videos.size() - 1);
                            if (videos.isEmpty()) {
                                releasePlayer();
                                segments.clear();
                                updateSegmentsUi();
                                info.setText("No quedan videos pendientes.");
                            } else {
                                openCurrent(true);
                            }
                        }
                    });
                } catch (final Exception e) {
                    runOnUiThread(new Runnable() {
                        @Override public void run() {
                            exporting = false;
                            info.setText("Error exportando: " + e.getMessage());
                            showToast("Error exportando");
                        }
                    });
                }
            }
        }, "videocortes-export").start();
    }

    private void trimVideo(Uri inputUri, Uri outputUri, long startUs, long endUs) throws Exception {
        MediaExtractor extractor = new MediaExtractor();
        ParcelFileDescriptor pfd = null;
        MediaMuxer muxer = null;
        try {
            extractor.setDataSource(this, inputUri, null);
            pfd = getContentResolver().openFileDescriptor(outputUri, "rw");
            if (pfd == null) throw new Exception("Sin descriptor de salida");
            muxer = new MediaMuxer(pfd.getFileDescriptor(), MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4);
            int rotation = readRotation(inputUri);
            if (rotation == 90 || rotation == 180 || rotation == 270) muxer.setOrientationHint(rotation);

            HashMap<Integer, Integer> trackMap = new HashMap<>();
            int maxInputSize = 1024 * 1024;
            for (int i = 0; i < extractor.getTrackCount(); i++) {
                MediaFormat format = extractor.getTrackFormat(i);
                String mime = format.containsKey(MediaFormat.KEY_MIME) ? format.getString(MediaFormat.KEY_MIME) : "";
                if (mime == null || !(mime.startsWith("video/") || mime.startsWith("audio/"))) continue;
                if (format.containsKey(MediaFormat.KEY_MAX_INPUT_SIZE)) {
                    maxInputSize = Math.max(maxInputSize, format.getInteger(MediaFormat.KEY_MAX_INPUT_SIZE));
                }
                int dst = muxer.addTrack(format);
                trackMap.put(i, dst);
                extractor.selectTrack(i);
            }
            if (trackMap.isEmpty()) throw new Exception("Sin pistas de video/audio compatibles");
            muxer.start();
            extractor.seekTo(startUs, MediaExtractor.SEEK_TO_PREVIOUS_SYNC);

            ByteBuffer buffer = ByteBuffer.allocate(Math.min(Math.max(maxInputSize, 1024 * 1024), 8 * 1024 * 1024));
            android.media.MediaCodec.BufferInfo sampleInfo = new android.media.MediaCodec.BufferInfo();
            long firstSampleUs = -1;
            while (true) {
                int track = extractor.getSampleTrackIndex();
                if (track < 0) break;
                Integer dstTrack = trackMap.get(track);
                if (dstTrack == null) {
                    extractor.advance();
                    continue;
                }
                long sampleTime = extractor.getSampleTime();
                if (sampleTime < 0 || sampleTime > endUs) break;
                int size = extractor.readSampleData(buffer, 0);
                if (size < 0) break;
                if (firstSampleUs < 0) firstSampleUs = sampleTime;
                sampleInfo.set(0, size, Math.max(0, sampleTime - firstSampleUs), extractor.getSampleFlags());
                muxer.writeSampleData(dstTrack, buffer, sampleInfo);
                extractor.advance();
            }
        } finally {
            try { extractor.release(); } catch (Exception ignored) {}
            try { if (muxer != null) { muxer.stop(); muxer.release(); } } catch (Exception ignored) {}
            try { if (pfd != null) pfd.close(); } catch (Exception ignored) {}
        }
    }

    private int readRotation(Uri uri) {
        MediaMetadataRetriever r = new MediaMetadataRetriever();
        try {
            r.setDataSource(this, uri);
            String raw = r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION);
            return raw == null ? 0 : Integer.parseInt(raw);
        } catch (Exception e) {
            return 0;
        } finally {
            try { r.release(); } catch (Exception ignored) {}
        }
    }

    private void createTextFile(String parentDocId, String name, String text) throws Exception {
        Uri parent = DocumentsContract.buildDocumentUriUsingTree(treeUri, parentDocId);
        Uri doc = DocumentsContract.createDocument(getContentResolver(), parent, "text/plain", name);
        if (doc == null) throw new Exception("No se pudo crear marcador");
        OutputStream out = getContentResolver().openOutputStream(doc, "wt");
        if (out == null) throw new Exception("No se pudo escribir marcador");
        try {
            out.write(text.getBytes("UTF-8"));
        } finally {
            out.close();
        }
    }

    private String uniqueName(String parentDocId, String name) {
        Set<String> names = listNames(parentDocId);
        if (!names.contains(name)) return name;
        int dot = name.lastIndexOf('.');
        String base = dot > 0 ? name.substring(0, dot) : name;
        String ext = dot > 0 ? name.substring(dot) : "";
        for (int i = 2; i < 1000; i++) {
            String candidate = base + "-" + i + ext;
            if (!names.contains(candidate)) return candidate;
        }
        return base + "-" + System.currentTimeMillis() + ext;
    }

    private Set<String> listNames(String parentDocId) {
        Set<String> names = new HashSet<>();
        Cursor c = null;
        try {
            Uri childrenUri = DocumentsContract.buildChildDocumentsUriUsingTree(treeUri, parentDocId);
            c = getContentResolver().query(childrenUri, new String[] {
                    DocumentsContract.Document.COLUMN_DISPLAY_NAME
            }, null, null, null);
            if (c != null) {
                while (c.moveToNext()) names.add(c.getString(0));
            }
        } catch (Exception ignored) {
        } finally {
            if (c != null) c.close();
        }
        return names;
    }

    private String baseName(String name) {
        int dot = name.lastIndexOf('.');
        return dot > 0 ? name.substring(0, dot) : name;
    }

    private String safeName(String name) {
        return name.replaceAll("[\\\\/:*?\"<>|]", "_");
    }

    private String fmt(long ms) {
        long total = Math.max(0, ms);
        long hours = total / 3600000;
        long minutes = (total / 60000) % 60;
        long seconds = (total / 1000) % 60;
        long millis = total % 1000;
        return String.format(Locale.US, "%02d.%02d.%02d.%03d", hours, minutes, seconds, millis);
    }

    private void updateInfo() {
        if (exporting || videos.isEmpty()) return;
        VideoItem item = videos.get(index);
        long pos = currentPosition();
        info.setText((index + 1) + "/" + videos.size() + "  " + item.name
                + "\nP " + (isPlaying() ? "pausa" : "play")
                + "  Pos " + fmt(pos)
                + "  I " + fmt(cutStartMs)
                + "  Segmentos " + segments.size());
    }

    private void updateSegmentsUi() {
        if (segmentBar != null) {
            segmentBar.setState(durationMs, currentPosition(), cutStartMs, cutEndMs, segments);
        }
        if (positionSeek != null && !userScrubbing) {
            int progress = durationMs <= 0 ? 0 : (int) Math.min(10000, (currentPosition() * 10000L) / durationMs);
            positionSeek.setProgress(progress);
        }
        if (segmentText == null) return;
        if (segments.isEmpty()) {
            segmentText.setText("Segmentos: ninguno. Pulsa I al inicio y O al final.");
            return;
        }
        StringBuilder sb = new StringBuilder("Segmentos: ");
        for (int i = 0; i < segments.size(); i++) {
            Segment s = segments.get(i);
            if (i > 0) sb.append("   ");
            sb.append(i + 1).append(") ").append(fmt(s.startMs)).append("-").append(fmt(s.endMs));
        }
        segmentText.setText(sb.toString());
    }

    private void showToast(final String text) {
        runOnUiThread(new Runnable() {
            @Override public void run() {
                Toast.makeText(MainActivity.this, text, Toast.LENGTH_SHORT).show();
            }
        });
    }

    @Override
    public boolean dispatchKeyEvent(KeyEvent event) {
        if (event.getAction() == KeyEvent.ACTION_DOWN) {
            int code = event.getKeyCode();
            if (code == KeyEvent.KEYCODE_P || code == KeyEvent.KEYCODE_SPACE) { togglePlay(); return true; }
            if (code == KeyEvent.KEYCODE_I) { markIn(); return true; }
            if (code == KeyEvent.KEYCODE_O) { markOut(); return true; }
            if (code == KeyEvent.KEYCODE_E) { exportCut(); return true; }
            if (code == KeyEvent.KEYCODE_N) { nextVideo(); return true; }
        }
        return super.dispatchKeyEvent(event);
    }

    private static class Doc {
        String id;
        String name;
        String mime;
        long modified;
    }

    private static class VideoItem {
        String name;
        String base;
        Uri uri;
        String parentDocId;
        long modified;
    }

    private static class Segment {
        final long startMs;
        final long endMs;

        Segment(long startMs, long endMs) {
            this.startMs = startMs;
            this.endMs = endMs;
        }
    }

    private static class SegmentBar extends View {
        private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
        private long durationMs;
        private long positionMs;
        private long currentStartMs;
        private long currentEndMs;
        private ArrayList<Segment> segments = new ArrayList<>();

        SegmentBar(Activity activity) {
            super(activity);
            setBackgroundColor(0xff111827);
        }

        void setState(long durationMs, long positionMs, long currentStartMs, long currentEndMs, ArrayList<Segment> segments) {
            this.durationMs = Math.max(1, durationMs);
            this.positionMs = Math.max(0, positionMs);
            this.currentStartMs = Math.max(0, currentStartMs);
            this.currentEndMs = currentEndMs;
            this.segments = new ArrayList<>(segments);
            invalidate();
        }

        @Override
        protected void onDraw(Canvas canvas) {
            super.onDraw(canvas);
            int w = getWidth();
            int h = getHeight();
            paint.setColor(0xff374151);
            canvas.drawRect(0, h / 2f - 4, w, h / 2f + 4, paint);
            paint.setColor(0xff22c55e);
            for (Segment s : segments) {
                float left = (s.startMs / (float) durationMs) * w;
                float right = (s.endMs / (float) durationMs) * w;
                canvas.drawRect(left, dpLocal(4), Math.max(left + 3, right), h - dpLocal(4), paint);
            }
            if (currentEndMs <= currentStartMs) {
                paint.setColor(0xfffacc15);
                float x = (currentStartMs / (float) durationMs) * w;
                canvas.drawRect(x - 2, 0, x + 2, h, paint);
            }
            paint.setColor(0xffef4444);
            float pos = (positionMs / (float) durationMs) * w;
            canvas.drawRect(pos - 2, 0, pos + 2, h, paint);
        }

        private int dpLocal(int v) {
            return (int) (v * getResources().getDisplayMetrics().density + 0.5f);
        }
    }
}
