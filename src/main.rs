extern crate hound;
extern crate bmp;
extern crate rustfft;
extern crate spmc;

use rustfft::FFTplanner;
use std::cmp;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use std::sync::mpsc;
use std::thread;
use std::io;
use std::sync::atomic;
use std::sync;
use std::io::Write;

const FFT_SIZE: usize = 1024*8;
const NORM_SIZE: usize = (FFT_SIZE/2);
const NUM_WORKERS: usize = 16;

fn clear(vec: &mut Vec<Complex<f32>>) {
    for i in vec {
        i.re= 0_f32;
        i.im = 0_f32;
    }
}   

fn freq_to_bin(freq: u32, sampleRate: u32) -> usize {
    (freq as f32 * NORM_SIZE as f32 / sampleRate as f32).round() as usize
}

enum PrintJob {
    Print(u32, Vec<Complex<f32>>, u32),
    Done
}

fn main() {

    let reader = hound::WavReader::open("sample.wav").unwrap();
    let sampleSize = reader.spec().sample_rate;

    // how many times we'll need to stdf 
    let iterations = reader.duration() / reader.spec().channels as u32  / FFT_SIZE as u32;

    /*
     * main thread spawns a printing thread and NUM_WORERS stdf threads.
     * main thread pushes interation indicies for worker threads (jobProd)
     * workers push printing jobs (printProd)
     */
    let (jobProd, jobRecv) = spmc::channel();
    let (printProd, printRecv) = mpsc::channel();

    let printThread = thread::spawn(move || {

        let widthPerIter = (1_f32 + (FFT_SIZE as f32/ 1024_f32).round()) as u32;
        println!("{}", widthPerIter);
        let mut img = bmp::Image::new(widthPerIter * iterations, NORM_SIZE as u32);

        for msg in printRecv {
            match msg {
                PrintJob::Print(iteration, data, sampleRate) => {
                    let minFreq = 20;
                    let maxFreq = 8000;
                    
                    for i in freq_to_bin(minFreq, sampleRate)..freq_to_bin(maxFreq, sampleRate) {

                        let norm = data[i].norm_sqr().sqrt() / NORM_SIZE as f32;
                        let freq = i as f32 * sampleRate as f32 / FFT_SIZE as f32;

                        // gamma correction!
                        let grayscale = (f32::powf(norm, 0.45_f32) * 255_f32) as u8;
                        
                        // uncorrected blue channel paints louder sections purple-ish
                        let mut color =  bmp::Pixel::new(grayscale, 0, (norm * 255_f32) as u8);

                        for x in 0..widthPerIter {
                            img.set_pixel((iteration * widthPerIter) + x, (NORM_SIZE- i - 1) as u32, color); 
                        }
                    }
                }
                PrintJob::Done => break
            }
        }
        let _ = img.save("out.bmp");
    });
    
    for iteration in 0..iterations {
        jobProd.send(iteration);
    }

    let mut workers: Vec<_> = Vec::new();

    let processed = sync::Arc::new(atomic::AtomicUsize::new(0));

    for i in 0..NUM_WORKERS {
        let jobRecv = jobRecv.clone();
        let printProd = printProd.clone();
        let processed = processed.clone();

        workers.push(thread::spawn(move || {
         
            let mut reader = hound::WavReader::open("sample.wav").unwrap();
            let spec = &reader.spec();

            let mut planner = FFTplanner::new(false);
            let fft = planner.plan_fft(FFT_SIZE);

            loop {
                let mut input:  Vec<Complex<f32>> = vec![Complex::zero(); FFT_SIZE];

                match jobRecv.try_recv() {
                    Ok(iteration) => {
                        reader.seek(iteration * spec.channels as u32 * FFT_SIZE as u32);
                        let mut iter = reader.samples::<i16>();
                        let mut output: Vec<Complex<f32>> = vec![Complex::zero(); FFT_SIZE];

                        for i in 0..FFT_SIZE {
                            if let Some(elem) = input.get_mut(i)
                            {
                                let sample = match iter.next() {
                                    Some(val) => match val { Result::Ok(val2) => val2 as f32 / std::i16::MAX as f32, Result::Err(_) => break }
                                    None => break
                                };

                                match spec.channels {
                                    1 =>  elem.re = sample,
                                    2 => {
                                        let right = match iter.next() {
                                            Some(val) => match val { Result::Ok(val2) => val2 as f32 / std::i16::MAX as f32, Result::Err(_) => break }
                                            None => break
                                        };
                                        elem.re = (sample + right) * 0.5_f32;
                                    }
                                    _ => panic!("Unsupported channel number: {}", spec.channels)
                                };

                                // hanning window
                                elem.re *= 0.5_f32 * (1_f32 - ((2_f32 * std::f32::consts::PI * i as f32) / (FFT_SIZE- 1) as f32));
                            }
                        }

                        fft.process(&mut input, &mut output);
                        
                        printProd.send(PrintJob::Print(iteration, output, spec.sample_rate));

                        clear(&mut input);
                        processed.fetch_add(1, atomic::Ordering::Release);
                    }
                    Err(_) => break
                }
            }
        }));
    }

    let workers = workers;

    loop {
        let count = processed.load(atomic::Ordering::Acquire);
        print!("\r{}/{}", count, iterations);
        io::stdout().flush().ok().expect("Could not flush stdout");

        match count.cmp(&(iterations as usize)) {
            cmp::Ordering::Equal | cmp::Ordering::Greater => break,
            _ => std::thread::sleep(std::time::Duration::from_millis(100))
        };
    }

    for worker in workers {
        worker.join().unwrap();
    }

    printProd.send(PrintJob::Done).unwrap();
    printThread.join().unwrap();
}
