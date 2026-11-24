use std::sync::Mutex;
use std::thread;

/// Run `f(row_index, row)` over every row of `out`, spread across all CPU cores.
/// Rows are handed out in small bands so uneven content still balances well.
pub fn par_rows<T: Send>(out: &mut [T], row_len: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    if row_len == 0 || out.is_empty() {
        return;
    }
    let rows = out.len() / row_len;
    let n = thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(rows).max(1);
    let band = 4usize;
    let queue: Mutex<Vec<(usize, &mut [T])>> = Mutex::new(out.chunks_mut(row_len * band).enumerate().collect());
    thread::scope(|s| {
        for _ in 0..n {
            s.spawn(|| loop {
                let job = queue.lock().unwrap().pop();
                let Some((ci, chunk)) = job else { break };
                for (ri, row) in chunk.chunks_mut(row_len).enumerate() {
                    f(ci * band + ri, row);
                }
            });
        }
    });
}
