{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}
import Criterion.Main
    ( defaultMain, bench, bgroup, whnf, Benchmark )
import Data.Array.Accelerate
import Data.Accelerate.TensorFlow.Execute (TensorFlow)
import Data.Array.Accelerate.Interpreter (Interpreter)
import Data.Accelerate.TensorFlow.Operation ()
import Data.Accelerate.TensorFlow.Desugar ()
import Data.Accelerate.TensorFlow.Kernel ()
import Data.Array.Accelerate.Pretty.Schedule ()
import Data.Array.Accelerate.AST.Schedule.Sequential ()
import Data.Array.Accelerate.Pretty.Schedule.Sequential ()
import Prelude hiding (uncurry, curry, (^^), (^), lcm, gcd, (||), not, iterate, scaleFloat, isNaN, isInfinite, isDenormalized, isNegativeZero, atan2, isIEEE, significand, exponent, encodeFloat, decodeFloat, floatRange, floatDigits, floatRadix, properFraction, floor, ceiling, round, toRational, compare, min, (/=), (==), scanr1, scanr, scanl1, scanl, Ord, maximum, minimum, product, or, and, any, all, max, odd, even, reverse, Num, drop, take, tail, init, replicate, unzip3, unzip, zip, zipWith3, zip3, (<=), (>), filter, (&&), (>=), subtract, (<), truncate, fromIntegral, map, (+))
import Lens.Micro ( Lens', lens )
import Data.Array.Accelerate.Data.Ratio (Ratio)

-- This file contains the benchmarks for test cases that passed.
-- To enable benchmarking, the contents of ../test/Main.hs were copied, and the failing / unsupported tests were commented out

type Stencil5x1 a = (Stencil3 a, Stencil5 a, Stencil3 a)
type Stencil1x5 a = (Stencil3 a, Stencil3 a, Stencil3 a, Stencil3 a, Stencil3 a)

assertAcc :: a -> a
assertAcc = id

-- Our benchmark harness.
main :: IO ()
main = defaultMain tests

testCase :: Arrays a => String -> Acc a -> Benchmark
testCase s acc = -- bench (s Prelude.++ " TensorFlow") $ whnf (run @TensorFlow) acc
                 bench (s Prelude.++ " Interpreter") $ whnf (run @Interpreter) acc
                 -- swap lines above to switch backends.

-- Lenses
-- ------
--
-- Imported from `lens-accelerate` (which provides more general Field instances)
--
_1 :: forall sh. Elt sh => Lens' (Exp (sh:.Int)) (Exp Int)
_1 = lens (\(ix :: Exp (sh :. Int)) -> let _  :. x = unlift ix :: Exp sh :. Exp Int in x)
          (\ix x -> let sh :. _ = unlift ix :: Exp sh :. Exp Int in lift (sh :. x))

_2 :: forall sh. Elt sh => Lens' (Exp (sh:.Int:.Int)) (Exp Int)
_2 = lens (\(ix :: Exp (sh :. Int :. Int))   -> let _  :. y :. _ = unlift ix :: Exp sh :. Exp Int :. Exp Int in y)
          (\ix y -> let sh :. _ :. x = unlift ix :: Exp sh :. Exp Int :. Exp Int in lift (sh :. y :. x))

_3 :: forall sh. Elt sh => Lens' (Exp (sh:.Int:.Int:.Int)) (Exp Int)
_3 = lens (\(ix :: Exp (sh :. Int :. Int :. Int))   -> let _  :. z :. _ :. _ = unlift ix :: Exp sh :. Exp Int :. Exp Int :. Exp Int in z)
          (\ix z -> let sh :. _ :. y :. x = unlift ix :: Exp sh :. Exp Int :. Exp Int :. Exp Int in lift (sh :. z :. y :. x))

tests :: [Benchmark]
tests =  [ tAccelerateArrayLanguage,
           tAccelerateExpressionLanguage ]

-- | Test tree corresponding to section "The Accelerate Array Language" of the Accelerate documentation.
tAccelerateArrayLanguage :: Benchmark
tAccelerateArrayLanguage = bgroup "The Accelerate Array Language"
  [ tConstruction,
    tComposition,
    tElementWiseOperations,
    tModifyingArrays
    -- tFolding, FAILS 
    -- tScans,
    -- tStencils
  ]
  where tConstruction :: Benchmark
        tConstruction = bgroup "Construction"
          [
            tIntroduction,
            tInitialisation,
            tEnumeration,
            tConcatenation,
            tExpansion
          ]
          where tIntroduction = bgroup "Introduction"
                  [ tUse,
                    tUnit
                  ]
                  where tUse = bgroup "use"
                          [ testCase "use vec" $ assertAcc $ use (fromList (Z:.10) [0..] :: Vector Int64),
                            testCase "use mat" $ assertAcc $ use (fromList (Z:.5:.10) [0..] :: Matrix Int64),
                            testCase "use tup" $ assertAcc $ use (fromList (Z:.10) [0..] :: Vector Int64, fromList (Z:.5:.10) [0..] :: Matrix Int64)
                          ]
                        tUnit = bgroup "unit"
                          [ testCase "unit 1" $ assertAcc $ unit (constant 1 :: Exp Int64),
                            testCase "unit (1, 2)" $ assertAcc $ unit (constant (1, 2) :: Exp (Int64, Int64))
                          ]
                tInitialisation = bgroup "Initialisation"
                  [ tGenerate,
                    tFill
                  ]
                  where tGenerate = bgroup "Generate"
                          [ testCase "generate 1.2s" $ assertAcc (generate (I1 3) (const 1.2) :: Acc (Array DIM1 Float)),
                            testCase "generate [1..]" $ assertAcc (generate (I1 10) (\(I1 i) -> fromIntegral $ i + 1) :: Acc (Array DIM1 Int64)),
                            testCase "generate even" $ assertAcc (generate (I1 5) (\(I1 i) -> fromIntegral $ i `mod` 2) :: Acc (Array DIM1 Int64))
                          ]
                        tFill = bgroup "Fill"
                          [ testCase "fill 1.2s" $ assertAcc (fill (constant (Z :. 3)) 1.2 :: Acc (Array DIM1 Float))
                          ]
                tEnumeration = bgroup "Enumeration"
                  [ tEnumFromN,
                    tEnumFromStepN
                  ]
                  where tEnumFromN = bgroup "enumFromN"
                          [ testCase "x+1" $ assertAcc (enumFromN (constant (Z:.5:.10)) 0 :: Acc (Array DIM2 Int64))
                          ]
                        tEnumFromStepN = bgroup "enumFromStepN"
                          [ testCase "x+ny" $ assertAcc (enumFromStepN (constant (Z:.5:.10)) 0 0.5 :: Acc (Array DIM2 Float))
                          ]
                tConcatenation = bgroup "Concatenation"
                  [ -- tPlusPlus, FAILS
                    -- tConcatOn
                  ]
                  where tPlusPlus = bgroup "++"
                          [ testCase "++" $ assertAcc (use (fromList (Z:.5:.10) [0..]) Data.Array.Accelerate.++ use (fromList (Z:.10:.3) [0..]) :: Acc (Array DIM2 Int64))
                          ]
                        tConcatOn = let
                              m1 = fromList (Z:.5:.10) [0..] :: Matrix Int64
                              m2 = fromList (Z:.10:.5) [0..] :: Matrix Int64
                            in bgroup "concatOn"
                          [ testCase "concatOn _1" $ assertAcc (concatOn _1 (use m1) (use m2)),
                            testCase "concatOn _2" $ assertAcc (concatOn _2 (use m1) (use m2))
                          ]
                tExpansion = bgroup "Expansion"
                  [ -- tExpand FAILS
                  ]
                  where tExpand = let primes :: Exp Int -> Acc (Vector Int)
                                      primes n = afst loop
                                        where
                                          c0    = unit 2
                                          a0    = use $ fromList (Z:.0) []
                                          limit = truncate (sqrt (fromIntegral (n+1) :: Exp Float))
                                          loop  = awhile
                                                    (\(T2 _   c) -> map (< n+1) c)
                                                    (\(T2 old c) ->
                                                      let c1 = the c
                                                          c2 = c1 < limit ? ( c1*c1, n+1 )
                                                          --
                                                          sieves =
                                                            let sz p    = (c2 - p) `quot` p
                                                                get p i = (2+i)*p
                                                            in
                                                            map (subtract c1) (expand sz get old)
                                                          --
                                                          new =
                                                            let m     = c2-c1
                                                                put i = let s = sieves ! i
                                                                        in s >= 0 && s < m ? (Just_ (I1 s), Nothing_)
                                                            in
                                                            afst
                                                              $ filter (> 0)
                                                              $ permute const (enumFromN (I1 m) c1) put
                                                              $ fill (shape sieves) 0
                                                      in
                                                      T2 (old Data.Array.Accelerate.++ new) (unit c2))
                                                    (T2 a0 c0)
                          in bgroup "expand"
                          [ testCase "expand" $ assertAcc $ primes 100
                          ]

        tComposition :: Benchmark
        tComposition = bgroup "Composition"
          [ tFlowControl,
            tControllingExecution
          ]
          where tFlowControl = bgroup "Flow Control"
                  [ tIacond,
                    tAcond,
                    tAWhile,
                    tIfThenElse
                  ]
                  where tIacond = bgroup "infix acond (?|)"
                          [ testCase "(?|) true" $ assertAcc (True_ ?| (use $ fromList (Z:.5:.10) [0..], use $ fromList (Z:.5:.10) [5..]) :: Acc (Array DIM2 Int64)),
                            testCase "(?|) false" $ assertAcc (False_ ?| (use $ fromList (Z:.5:.10) [0..], use $ fromList (Z:.5:.10) [5..]) :: Acc (Array DIM2 Int64))
                          ]
                        tAcond = bgroup "acond"
                          [ testCase "acond true" $ assertAcc (acond True_ (use $ fromList (Z:.5:.10) [0..]) (use $ fromList (Z:.5:.10) [5..]) :: Acc (Array DIM2 Int64)),
                            testCase "acond false" $ assertAcc (acond False_ (use $ fromList (Z:.5:.10) [0..]) (use $ fromList (Z:.5:.10) [5..]) :: Acc (Array DIM2 Int64))
                          ]
                        tAWhile = bgroup "awhile"
                          [ testCase "awhile" $ assertAcc (awhile (\x -> unit $ x!I1 0 <= 10) (map (+ 1)) (use $ fromList (Z:.10) [0..] :: Acc (Array DIM1 Int64)) :: Acc (Array DIM1 Int64))
                          ]
                        tIfThenElse = bgroup "if then else"
                          [ testCase "if then else" $ assertAcc (ifThenElse (constant True) (use $ fromList (Z:.5:.10) [0..]) (use $ fromList (Z:.5:.10) [1..]) :: Acc (Array DIM2 Int64))
                          ]
                tControllingExecution = bgroup "Controlling Execution"
                  [ tPipeline
                    -- tCompute FAILS
                  ]
                  where tPipeline = bgroup "pipeline (>->)"
                          [ testCase "pipeline (>->)" $ assertAcc ((>->) (map (+1)) (map (/ 2)) (use (fromList (Z:.5:.10) [1..])) :: Acc (Array DIM2 Float))
                          ]
                        tCompute = let
                          loop :: Exp Int -> Exp Int
                          loop ticks = let clockRate = 900000   -- kHz
                                      in  while (\i -> i < clockRate * ticks) (+1) 0
                          in bgroup "compute"
                          [ testCase "compute" $ assertAcc $ zip3
                              (compute $ map loop (use $ fromList (Z:.1) [10]))
                              (compute $ map loop (use $ fromList (Z:.1) [10]))
                              (compute $ map loop (use $ fromList (Z:.1) [10]))
                          ]
                        x = let
                              vec    = use $ fromList (Z:.5) [0..] :: Acc (Vector Int)
                              f      = (\x -> (x + 1) * 2) :: Exp Int -> Exp Int
                            in map f vec :: Acc (Vector Int)

        tElementWiseOperations :: Benchmark
        tElementWiseOperations = bgroup "Element-wise operations"
          [ tIndexing,
            tMapping,
            tZipping,
            tUnzipping
          ]
          where tIndexing = bgroup "Indexing"
                  [ tIndexed
                  ]
                  where tIndexed = bgroup "indexed"
                          [ testCase "indexed vec" $ assertAcc (indexed (use (fromList (Z:.5) [0..] :: Vector Float))),
                            testCase "indexed mat" $ assertAcc (indexed (use (fromList (Z:.3:.4) [0..] :: Matrix Float)))
                          ]
                tMapping = bgroup "Mapping"
                  [ tMap,
                    tIMap
                  ]
                  where tMap = bgroup "map"
                          [ testCase "map" $ assertAcc (map (+1) (use (fromList (Z:.5) [0..] :: Vector Float)))
                          ]
                        tIMap = bgroup "imap"
                          [ testCase "imap" $ assertAcc (imap (\(I1 i) x -> x + fromIntegral i) (use (fromList (Z:.5) [0..] :: Vector Int64)))
                          ]
                tZipping = bgroup "Zipping"
                  [ tZipWith,
                    tZipWith3,
                    tZipWith4,
                    tZipWith5,
                    tZipWith6,
                    tZipWith7,
                    tZipWith8,
                    tZipWith9,
                    tIZipWith,
                    tIZipWith3,
                    tIZipWith4,
                    tIZipWith5,
                    tIZipWith6,
                    -- tIZipWith7, FAILS
                    -- tIZipWith8, 
                    tIZipWith9,
                    tZip,
                    tZip3,
                    tZip4,
                    tZip5,
                    tZip6,
                    tZip7,
                    tZip8
                    -- tZip9 FAILS
                  ]
                  where vec = use (fromList (Z:.5) [1..] :: Vector Int64)
                        tZipWith = bgroup "zipWith"
                          [ testCase "zipWith" $ assertAcc (Data.Array.Accelerate.zipWith (+) vec vec)
                          ]
                        tZipWith3 = bgroup "zipWith3"
                          [ testCase "zipWith3" $ assertAcc (zipWith3 (\a b c -> a + b + c) vec vec vec)
                          ]
                        tZipWith4 = bgroup "zipWith4"
                          [ testCase "zipWith4" $ assertAcc (zipWith4 (\a b c d -> a + b + c + d) vec vec vec vec)
                          ]
                        tZipWith5 = bgroup "zipWith5"
                          [ testCase "zipWith5" $ assertAcc (zipWith5 (\a b c d e -> a + b + c + d + e) vec vec vec vec vec)
                          ]
                        tZipWith6 = bgroup "zipWith6"
                          [ testCase "zipWith6" $ assertAcc (zipWith6 (\a b c d e f -> a + b + c + d + e + f) vec vec vec vec vec vec)
                          ]
                        tZipWith7 = bgroup "zipWith7"
                          [ testCase "zipWith7" $ assertAcc (zipWith7 (\a b c d e f g-> a + b + c + d + e + f + g) vec vec vec vec vec vec vec)
                          ]
                        tZipWith8 = bgroup "zipWith8"
                          [ testCase "zipWith8" $ assertAcc (zipWith8 (\a b c d e f g h -> a + b + c + d + e + f + g + h) vec vec vec vec vec vec vec vec)
                          ]
                        tZipWith9 = bgroup "zipWith9"
                          [ testCase "zipWith9" $ assertAcc (zipWith9 (\a b c d e f g h i -> a + b + c + d + e + f + g + h + i) vec vec vec vec vec vec vec vec vec)
                          ]
                        tIZipWith = bgroup "izipWith"
                          [ testCase "izipWith" $ assertAcc (izipWith (\(I1 i) a b -> a + b + fromIntegral i) vec vec)
                          ]
                        tIZipWith3 = bgroup "izipWith3"
                          [ testCase "izipWith3" $ assertAcc (izipWith3 (\(I1 i) a b c -> a + b + c + fromIntegral i) vec vec vec)
                          ]
                        tIZipWith4 = bgroup "izipWith4"
                          [ testCase "izipWith4" $ assertAcc (izipWith4 (\(I1 i) a b c d -> a + b + c + d + fromIntegral i) vec vec vec vec)
                          ]
                        tIZipWith5 = bgroup "izipWith5"
                          [ testCase "izipWith5" $ assertAcc (izipWith5 (\(I1 i) a b c d e -> a + b + c + d + e + fromIntegral i) vec vec vec vec vec)
                          ]
                        tIZipWith6 = bgroup "izipWith6"
                          [ testCase "izipWith6" $ assertAcc (izipWith6 (\(I1 i) a b c d e f -> a + b + c + d + e + f + fromIntegral i) vec vec vec vec vec vec)
                          ]
                        tIZipWith7 = bgroup "izipWith7"
                          [ testCase "izipWith7" $ assertAcc (izipWith7 (\(I1 i) a b c d e f g -> a + b + c + d + e + f + g + fromIntegral i) vec vec vec vec vec vec vec)
                          ]
                        tIZipWith8 = bgroup "izipWith8"
                          [ testCase "izipWith8" $ assertAcc (izipWith8 (\(I1 i) a b c d e f g h -> a + b + c + d + e + f + g + h + fromIntegral i) vec vec vec vec vec vec vec vec)
                          ]
                        tIZipWith9 = bgroup "izipWith9"
                          [ testCase "izipWith9" $ assertAcc (izipWith9 (\(I1 i) a b c d e f g h j -> a + b + c + d + e + f + g + h + j + fromIntegral i) vec vec vec vec vec vec vec vec vec)
                          ]
                        tZip = bgroup "zip"
                          [ testCase "zip" $ assertAcc (zip vec vec)
                          ]
                        tZip3 = bgroup "zip3"
                          [ testCase "zip3" $ assertAcc (zip3 vec vec vec)
                          ]
                        tZip4 = bgroup "zip4"
                          [ testCase "zip4" $ assertAcc (zip4 vec vec vec vec)
                          ]
                        tZip5 = bgroup "zip5"
                          [ testCase "zip5" $ assertAcc (zip5 vec vec vec vec vec)
                          ]
                        tZip6 = bgroup "zip6"
                          [ testCase "zip6" $ assertAcc (zip6 vec vec vec vec vec vec)
                          ]
                        tZip7 = bgroup "zip7"
                          [ testCase "zip7" $ assertAcc (zip7 vec vec vec vec vec vec vec)
                          ]
                        tZip8 = bgroup "zip8"
                          [ testCase "zip8" $ assertAcc (zip8 vec vec vec vec vec vec vec vec)
                          ]
                        tZip9 = bgroup "zip9"
                          [ testCase "zip9" $ assertAcc (zip9 vec vec vec vec vec vec vec vec vec)
                          ]
                tUnzipping = bgroup "Unzipping"
                  [
                    tUnzip,
                    tUnzip3,
                    tUnzip4,
                    tUnzip5,
                    tUnzip6,
                    tUnzip7,
                    tUnzip8,
                    tUnzip9
                  ]
                  where vec = use (fromList (Z:.5) [1..] :: Vector Int64)
                        tUnzip = bgroup "unzip"
                          [ --testCase "unzip" $ assertAcc2 (unzip $ zip vec vec :: (Acc (Vector Int64), Acc (Vector Int64)))
                          ]
                        tUnzip3 = bgroup "unzip3"
                          [ --testCase "unzip3" $ assertAcc3 (unzip3 $ zip3 vec vec vec :: (Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64)))
                          ]
                        tUnzip4 = bgroup "unzip4"
                          [ --testCase "unzip4" $ assertAcc4 (unzip4 $ zip4 vec vec vec vec :: (Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64)))
                          ]
                        tUnzip5 = bgroup "unzip5"
                          [ --testCase "unzip5" $ assertAcc5 (unzip5 $ zip5 vec vec vec vec vec :: (Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64)))
                          ]
                        tUnzip6 = bgroup "unzip6"
                          [ --testCase "unzip6" $ assertAcc6 (unzip6 $ zip6 vec vec vec vec vec vec :: (Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64)))
                          ]
                        tUnzip7 = bgroup "unzip7"
                          [ --testCase "unzip7" $ assertAcc7 (unzip7 $ zip7 vec vec vec vec vec vec vec :: (Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64)))
                          ]
                        tUnzip8 = bgroup "unzip7"
                          [ --testCase "unzip8" $ assertAcc8 (unzip8 $ zip8 vec vec vec vec vec vec vec vec :: (Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64)))
                          ]
                        tUnzip9 = bgroup "unzip9"
                          [ --testCase "unzip9" $ assertAcc9 (unzip9 $ zip9 vec vec vec vec vec vec vec vec vec :: (Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64), Acc (Vector Int64)))
                          ]

        tModifyingArrays :: Benchmark
        tModifyingArrays = bgroup "Modifying Arrays"
          [ tShapeManipulation,
            tReplication,
            tExtractingSubArrays,
            tPermutations
            -- tFiltering FAILS
          ]
          where tShapeManipulation = bgroup "Shape manipulation"
                  [ tReshape,
                    tFlatten
                  ]
                  where tReshape = bgroup "reshape"
                          [ testCase "reshape" $ assertAcc (reshape (constant (Z:.5:.5)) (use (fromList (Z:.25) [1..] :: Vector Int64)))
                          ]
                        tFlatten = bgroup "flatten"
                          [ testCase "flatten" $ assertAcc (flatten (use (fromList (Z:.5:.5) [1..] :: Matrix Int64)))
                          ]
                tReplication = bgroup "Replication"
                  [ tReplicate
                  ]
                  where tReplicate = let
                            vec = fromList (Z:.10) [0..] :: Vector Int
                            rep0 :: (Shape sh, Elt e) => Exp Int -> Acc (Array sh e) -> Acc (Array (sh :. Int) e)
                            rep0 n a = replicate (lift (Any :. n)) a
                            rep1 :: (Shape sh, Elt e) => Exp Int -> Acc (Array (sh :. Int) e) -> Acc (Array (sh :. Int :. Int) e)
                            rep1 n a = replicate (lift (Any :. n :. All)) a
                          in bgroup "replicate"
                          [ testCase "replicate 2d" $ assertAcc (replicate (constant (Z :. (4 :: Int) :. All)) (use vec)),
                            testCase "replicate 2d columns" $ assertAcc (replicate (lift (Z :. All :. (4::Int))) (use vec)),
                            testCase "replicate 2x1d, 3x 3d" $ assertAcc (replicate (constant (Z :. (2::Int) :. All :. (3::Int))) (use vec)),
                            testCase "rep0 1d" $ assertAcc $ rep0 10 (use $ fromList Z [42::Int]),
                            testCase "rep0 2d" $ assertAcc $ rep0 5 (use vec),
                            testCase "rep1" $ assertAcc $ rep1 5 (use vec)
                          ]
                tExtractingSubArrays = bgroup "Extracting subarrays"
                  [ tSlice,
                    tInit,
                    tTail,
                    tTake,
                    tDrop,
                    tSlit,
                    tInitOn,
                    tTailOn,
                    tTakeOn,
                    tDropOn,
                    tSlitOn
                  ]
                  where tSlice = let
                            mat = fromList (Z:.5:.10) [0..] :: Matrix Int
                            sl0 :: (Shape sh, Elt e) => Acc (Array (sh:.Int) e) -> Exp Int -> Acc (Array sh e)
                            sl0 a n = slice a (lift (Any :. n))
                            vec = fromList (Z:.10) [0..] :: Vector Int
                            sl1 :: (Shape sh, Elt e) => Acc (Array (sh:.Int:.Int) e) -> Exp Int -> Acc (Array (sh:.Int) e)
                            sl1 a n = slice a (lift (Any :. n :. All))
                            cube = fromList (Z:.3:.4:.5) [0..] :: Array DIM3 Int
                          in bgroup "slice"
                          [ testCase "slice mat 1d" $ assertAcc (slice (use mat) (constant (Z :. (2 :: Int) :. All))),
                            testCase "slice mat 0d" $ assertAcc (slice (use mat) (constant (Z :. 4 :. 2 :: DIM2))),
                            testCase "sl0 vec" $ assertAcc $ sl0 (use vec) 4,
                            testCase "sl0 mat" $ assertAcc $ sl0 (use mat) 4,
                            testCase "sl1 mat" $ assertAcc $ sl1 (use mat) 4,
                            testCase "sl1 cube" $ assertAcc $ sl1 (use cube) 2
                          ]
                        tInit = bgroup "init"
                          [ testCase "init mat" $ assertAcc (init (use (fromList (Z:.5:.10) [0..] :: Matrix Int64))),
                            testCase "init vec" $ assertAcc (init (use (fromList (Z:.10) [0..] :: Vector Int64)))
                          ]
                        tTail = bgroup "tail"
                          [
                            testCase "tail mat" $ assertAcc (tail (use (fromList (Z:.5:.10) [0..] :: Matrix Int64))),
                            testCase "tail vec" $ assertAcc (tail (use (fromList (Z:.10) [0..] :: Vector Int64)))
                          ]
                        tTake = bgroup "take"
                          [ testCase "take mat" $ assertAcc (take (constant 3) (use (fromList (Z:.5:.10) [0..] :: Matrix Int64))),
                            testCase "take vec" $ assertAcc (take (constant 3) (use (fromList (Z:.10) [0..] :: Vector Int64)))
                          ]
                        tDrop = bgroup "drop"
                          [ testCase "drop mat" $ assertAcc (drop (constant 3) (use (fromList (Z:.5:.10) [0..] :: Matrix Int64))),
                            testCase "drop vec" $ assertAcc (drop (constant 3) (use (fromList (Z:.10) [0..] :: Vector Int64)))
                          ]
                        tSlit = bgroup "slit"
                          [ testCase "slit mat" $ assertAcc (slit (constant 1) (constant 3) (use (fromList (Z:.5:.10) [0..] :: Matrix Int64))),
                            testCase "slit vec" $ assertAcc (slit (constant 1) (constant 3) (use (fromList (Z:.10) [0..] :: Vector Int64)))
                          ]
                        tInitOn = bgroup "initOn"
                          [ testCase "initOn mat" $ assertAcc (initOn _1 (use (fromList (Z:.5:.10) [0..] :: Matrix Int64))),
                            testCase "initOn vec" $ assertAcc (initOn _1 (use (fromList (Z:.10) [0..] :: Vector Int64)))
                          ]
                        tTailOn = bgroup "tailOn"
                          [ testCase "tailOn mat" $ assertAcc (tailOn _1 (use (fromList (Z:.5:.10) [0..] :: Matrix Int64))),
                            testCase "tailOn vec" $ assertAcc (tailOn _1 (use (fromList (Z:.10) [0..] :: Vector Int64)))
                          ]
                        tTakeOn = bgroup "takeOn"
                          [ testCase "takeOn mat" $ assertAcc (takeOn _1 1 (use (fromList (Z:.5:.10) [0..] :: Matrix Int64))),
                            testCase "takeOn vec" $ assertAcc (takeOn _1 1 (use (fromList (Z:.10) [0..] :: Vector Int64)))
                          ]
                        tDropOn = bgroup "dropOn"
                          [ testCase "dropOn mat" $ assertAcc (dropOn _1 1 (use (fromList (Z:.5:.10) [0..] :: Matrix Int64))),
                            testCase "dropOn vec" $ assertAcc (dropOn _1 1 (use (fromList (Z:.10) [0..] :: Vector Int64)))
                          ]
                        tSlitOn = bgroup "slitOn"
                          [ testCase "slitOn mat" $ assertAcc (slitOn _1 1 3 (use (fromList (Z:.5:.10) [0..] :: Matrix Int64))),
                            testCase "slitOn vec" $ assertAcc (slitOn _1 1 3 (use (fromList (Z:.10) [0..] :: Vector Int64)))
                          ]
                tPermutations = bgroup "Permutations"
                  [ tForwardPermutation,
                    tBackwardPermutation,
                    tSpecialisedPermutations
                  ]
                  where tForwardPermutation = bgroup "Forward Permutation (scatter)"
                          [ tPermute,
                            tScatter
                          ]
                          where tPermute = let
                                    histogram :: Acc (Vector Int) -> Acc (Vector Int)
                                    histogram xs =
                                      let zeros = fill (constant (Z :. 10)) 0
                                          ones  = fill (shape xs) 1
                                      in
                                      permute (+) zeros (\ix -> Just_ (I1 (xs!ix))) ones
                                    const2d :: Num a => Exp Int -> Acc (Matrix a)
                                    const2d n =
                                      let zeros = fill (I2 n n) 0
                                          ones  = fill (I2 n n) 1
                                      in
                                      permute const zeros (\(I2 i j) -> Just_ (I2 i j)) ones
                                  in bgroup "permute"
                                  [ testCase "histogram" $ assertAcc (histogram (use (fromList (Z :. 20) [0,0,1,2,1,1,2,4,8,3,4,9,8,3,2,5,5,3,1,2] :: Vector Int))),
                                    testCase "const2d" $ assertAcc (const2d 3 :: Acc (Matrix Int64))
                                  ]
                                tScatter = let
                                    to    = fromList (Z :. 6) [1,3,4,2,5,0] :: Vector Int
                                    input = fromList (Z :. 6) [1,9,6,4,4,2] :: Vector Int
                                  in bgroup "scatter"
                                  [ testCase "scatter" $ assertAcc $ scatter (use to) (fill (constant (Z:.6)) 0) (use input)
                                  ]
                        tBackwardPermutation = bgroup "Backward Permutation (gather)"
                          [ tBackpermute,
                            tGather
                          ]
                          where tBackpermute = let
                                    swap :: Exp DIM2 -> Exp DIM2
                                    swap = lift1 f
                                      where
                                        f :: Z :. Exp Int :. Exp Int -> Z :. Exp Int :. Exp Int
                                        f (Z:.y:.x) = Z :. x :. y
                                    mat = fromList (Z:.5:.10) [0..] :: Matrix Int
                                    mat' = use mat
                                  in bgroup "backpermute"
                                  [ testCase "backpermute swap" $ assertAcc $ backpermute (swap (shape mat')) swap mat'
                                  ]
                                tGather = let
                                    from  = fromList (Z :. 6) [1,3,6,2,5,4] :: Vector Int
                                    input = fromList (Z :. 7) [1,9,6,4,4,2,5] :: Vector Int
                                  in bgroup "gather"
                                  [ testCase "gather" $ assertAcc $ gather (use from) (use input)
                                  ]
                        tSpecialisedPermutations = bgroup "Specialised Permutations"
                          [
                            tReverse,
                            tTranspose,
                            tReverseOn,
                            tTransposeOn
                          ]
                          where tReverse = bgroup "reverse"
                                  [ testCase "reverse vec" $ assertAcc (reverse (use (fromList (Z :. 10) [0..] :: Vector Int64)))
                                  ]
                                tTranspose = bgroup "transpose"
                                  [ testCase "transpose mat" $ assertAcc (transpose (use (fromList (Z :. 5 :. 10) [0..] :: Matrix Int64)))
                                  ]
                                tReverseOn = bgroup "reverseOn"
                                  [ testCase "reverseOn mat" $ assertAcc (reverseOn _1 (use (fromList (Z:.5:.10) [0..] :: Matrix Int64))),
                                    testCase "reverseOn vec" $ assertAcc (reverseOn _1 (use (fromList (Z:.10) [0..] :: Vector Int64)))
                                  ]
                                tTransposeOn = bgroup "transposeOn"
                                  [ testCase "transposeOn mat" $ assertAcc (transposeOn _1 _1 (use (fromList (Z:.5:.10) [0..] :: Matrix Int64))),
                                    testCase "transposeOn vec" $ assertAcc (transposeOn _1 _1 (use (fromList (Z:.10) [0..] :: Vector Int64)))
                                  ]
                tFiltering = bgroup "Filtering"
                  [ -- tFilter, FAILS
                    -- tCompact FAILS
                  ]
                  where vec = fromList (Z :. 10) [1..10] :: Vector Int64
                        tFilter = bgroup "filter"
                          [ testCase "filter even" $ assertAcc (filter even (use vec)),
                            testCase "filter odd" $ assertAcc (filter odd (use vec))
                          ]
                        tCompact = bgroup "compact"
                          [ testCase "compact even" $ assertAcc (compact (map even $ use vec) $ use vec),
                            testCase "compact odd" $ assertAcc (compact (map odd $ use vec) $ use vec)
                          ]

        tFolding :: Benchmark
        tFolding = bgroup "Folding"
          [ tFold,
            tFold1,
            tFoldAll,
            tFold1All,
            tSegmentedReductions,
            tSpecialisedReductions
          ]
          where maximumSegmentSum
                      :: forall sh e. (Shape sh, Num e, Ord e)
                      => Acc (Array (sh :. Int) e)
                      -> Acc (Array sh e)
                maximumSegmentSum
                  = map (\(T4 x _ _ _) -> x)
                  . fold1 f
                  . map g
                  where
                    f :: (Num a, Ord a) => Exp (a,a,a,a) -> Exp (a,a,a,a) -> Exp (a,a,a,a)
                    f x y =
                      let T4 mssx misx mcsx tsx = x
                          T4 mssy misy mcsy tsy = y
                      in
                      T4 (mssx `max` (mssy `max` (mcsx+misy)))
                        (misx `max` (tsx+misy))
                        (mcsy `max` (mcsx+tsy))
                        (tsx+tsy)
                    --
                    g :: (Num a, Ord a) => Exp a -> Exp (a,a,a,a)
                    g x = let y = max x 0
                          in T4 y y y x
                tFold = bgroup "fold"
                  [ testCase "fold + 42" $ assertAcc (fold (+) 42 (use (fromList (Z:.5:.10) [0..] :: Matrix Int))),
                    testCase "fold maximumSegmentSum" $ assertAcc (maximumSegmentSum (use (fromList (Z:.10) [-2,1,-3,4,-1,2,1,-5,4,0] :: Vector Int)))
                  ]
                tFold1 = bgroup "fold1"
                  [ testCase "fold1 + 42" $ assertAcc (fold1 (+) (use (fromList (Z:.5:.10) [0..] :: Matrix Int))),
                    testCase "fold1 maximumSegmentSum" $ assertAcc (fold1 max (use (fromList (Z:.10) [-2,1,-3,4,-1,2,1,-5,4,0] :: Vector Int)))
                  ]
                tFoldAll = bgroup "foldAll"
                  [ testCase "foldAll + 42 vec" $ assertAcc (foldAll (+) 42 (use (fromList (Z:.10) [0..] :: Vector Float))),
                    testCase "foldAll + 0 mat" $ assertAcc (foldAll (+) 0 (use (fromList (Z:.5:.10) [0..] :: Matrix Float)))
                  ]
                tFold1All = bgroup "fold1All"
                  [ testCase "fold1All + 42 vec" $ assertAcc (fold1All (+) (use (fromList (Z:.10) [0..] :: Vector Float))),
                    testCase "fold1All + 0 mat" $ assertAcc (fold1All (+) (use (fromList (Z:.5:.10) [0..] :: Matrix Float)))
                  ]
                tSegmentedReductions = bgroup "Segmented Reductions"
                  [ tFoldSeg,
                    tFold1Seg
                  ]
                  where tFoldSeg = bgroup "foldSeg"
                          [ testCase "foldSeg vec" $ assertAcc (foldSeg (+) 0 (use (fromList (Z :. 10) [0..] :: Vector Int64)) (use (fromList (Z :. 10) [0..] :: Vector Int64))),
                            testCase "foldSeg mat" $ assertAcc (foldSeg (+) 0 (use (fromList (Z :. 5 :. 10) [0..] :: Matrix Int64)) (use (fromList (Z :. 5) [0..] :: Vector Int64)))
                          ]
                        tFold1Seg = bgroup "fold1Seg"
                          [ testCase "fold1Seg vec" $ assertAcc (fold1Seg (+) (use (fromList (Z :. 10) [0..] :: Vector Int64)) (use (fromList (Z :. 10) [0..] :: Vector Int64))),
                            testCase "fold1Seg mat" $ assertAcc (fold1Seg (+) (use (fromList (Z :. 5 :. 10) [0..] :: Matrix Int64)) (use (fromList (Z :. 5) [0..] :: Vector Int64)))
                          ]
                tSpecialisedReductions = bgroup "Specialised Reductions"
                  [ tAll,
                    tAny,
                    tAnd,
                    tOr,
                    tSum,
                    tProduct,
                    tMinimum,
                    tMaximum
                  ]
                  where vec = fromList (Z :. 10) [0..] :: Vector Int64
                        mat = fromList (Z :. 4 :. 10) [1,2,3,4,5,6,7,8,9,10,1,1,1,1,1,2,2,2,2,2,2,4,6,8,10,12,14,16,18,20,1,3,5,7,9,11,13,15,17,19] :: Matrix Int64
                        allTrue = fromList (Z :. 10) [True, True, True, True, True, True, True, True, True, True] :: Vector Bool
                        someFalse = fromList (Z :. 10) [True, True, False, True, True, True, True, True, True, True] :: Vector Bool
                        allFalse = fromList (Z :. 10) [False, False, False, False, False, False, False, False, False, False] :: Vector Bool
                        tAll = bgroup "all"
                          [ testCase "all vec" $ assertAcc (all even (use vec)),
                            testCase "all mat" $ assertAcc (all even (use mat))
                          ]
                        tAny = bgroup "any"
                          [ testCase "any vec" $ assertAcc (any even (use vec)),
                            testCase "any mat" $ assertAcc (any even (use mat))
                          ]
                        tAnd = bgroup "and"
                          [ testCase "and vec all True" $ assertAcc (and (use allTrue)),
                            testCase "and vec some False" $ assertAcc (and (use someFalse)),
                            testCase "and vec all False" $ assertAcc (and (use allFalse))
                          ]
                        tOr = bgroup "or"
                          [ testCase "or vec all True" $ assertAcc (or (use allTrue)),
                            testCase "or vec some False" $ assertAcc (or (use someFalse)),
                            testCase "or vec all False" $ assertAcc (or (use allFalse))
                          ]
                        tSum = bgroup "sum"
                          [ testCase "sum vec" $ assertAcc (Data.Array.Accelerate.sum (use vec)),
                            testCase "sum mat" $ assertAcc (Data.Array.Accelerate.sum (use mat))
                          ]
                        tProduct = bgroup "product"
                          [ testCase "product vec" $ assertAcc (product (use (fromList (Z :. 10) [1..] :: Vector Int64))),
                            testCase "product mat" $ assertAcc (product (use mat))
                          ]
                        tMinimum = bgroup "minimum"
                          [ testCase "minimum vec" $ assertAcc (minimum (use vec)),
                            testCase "minimum mat" $ assertAcc (minimum (use mat))
                          ]
                        tMaximum = bgroup "maximum"
                          [ testCase "maximum vec" $ assertAcc (maximum (use vec)),
                            testCase "maximum mat" $ assertAcc (maximum (use mat))
                          ]

        tScans :: Benchmark
        tScans = bgroup "Scans (prefix sums)"
          [ tScanl,
            tScanl1,
            tScanl',
            tScanr,
            tScanr1,
            tScanr',
            tPrescanl,
            tPostscanl,
            tPrescanr,
            tPostscanr,
            tSegmentedScans
          ]
          where vec = fromList (Z :. 10) [0,1,2,3,4,5,6,7,8,9] :: Vector Int64
                mat = fromList (Z :. 4 :. 10) [1,2,3,4,5,6,7,8,9,10,1,1,1,1,1,2,2,2,2,2,2,4,6,8,10,12,14,16,18,20,1,3,5,7,9,11,13,15,17,19] :: Matrix Int64
                tScanl = bgroup "scanl"
                  [ testCase "scanl vec" $ assertAcc (scanl (+) 10 (use vec)),
                    testCase "scanl mat" $ assertAcc (scanl (+) 42 (use mat))
                  ]
                tScanl1 = bgroup "scanl1"
                  [ testCase "scanl1 vec" $ assertAcc (scanl1 (+) (use vec)),
                    testCase "scanl1 mat" $ assertAcc (scanl1 (+) (use mat))
                  ]
                tScanl' = bgroup "scanl'"
                  [ testCase "scanl' vec" $ assertAcc (scanl' (+) 10 (use vec)),
                    testCase "scanl' mat" $ assertAcc (scanl' (+) 42 (use mat))
                  ]
                tScanr = bgroup "scanr"
                  [ testCase "scanr vec" $ assertAcc (scanr (+) 10 (use vec)),
                    testCase "scanr mat" $ assertAcc (scanr (+) 42 (use mat))
                  ]
                tScanr1 = bgroup "scanr1"
                  [ testCase "scanr1 vec" $ assertAcc (scanr1 (+) (use vec)),
                    testCase "scanr1 mat" $ assertAcc (scanr1 (+) (use mat))
                  ]
                tScanr' = bgroup "scanr'"
                  [ testCase "scanr' vec" $ assertAcc (scanr' (+) 10 (use vec)),
                    testCase "scanr' mat" $ assertAcc (scanr' (+) 42 (use mat))
                  ]
                tPrescanl = bgroup "prescanl"
                  [ testCase "prescanl vec" $ assertAcc (prescanl (+) 10 (use vec)),
                    testCase "prescanl mat" $ assertAcc (prescanl (+) 42 (use mat))
                  ]
                tPostscanl = bgroup "postscanl"
                  [ testCase "postscanl vec" $ assertAcc (postscanl (+) 10 (use vec)),
                    testCase "postscanl mat" $ assertAcc (postscanl (+) 42 (use mat))
                  ]
                tPrescanr = bgroup "prescanr"
                  [ testCase "prescanr vec" $ assertAcc (prescanr (+) 10 (use vec)),
                    testCase "prescanr mat" $ assertAcc (prescanr (+) 42 (use mat))
                  ]
                tPostscanr = bgroup "postscanr"
                  [ testCase "postscanr vec" $ assertAcc (postscanr (+) 10 (use vec)),
                    testCase "postscanr mat" $ assertAcc (postscanr (+) 42 (use mat))
                  ]
                tSegmentedScans = bgroup "segmented scans"
                  [ tScanlSeg,
                    tScanl1Seg,
                    tScanl'Seg,
                    tPrescanlSeg,
                    tPostscanlSeg,
                    tScanrSeg,
                    tScanr1Seg,
                    tScanr'Seg,
                    tPrescanrSeg,
                    tPostscanrSeg
                  ]
                  where seg = fromList (Z:.4) [1,4,0,3] :: Segments Int
                        mat = fromList (Z:.5:.10) [  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
                                10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                                30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                                40, 41, 42, 43, 44, 45, 46, 47, 48, 49] :: Matrix Int
                        tScanlSeg = bgroup "scanlSeg"
                          [ --testCase "scanlSeg mat" $ assertAcc (scanlSeg (+) 0 (use mat) (use seg))
                          ]
                        tScanl1Seg = bgroup "scanl1Seg"
                          [ --testCase "scanl1Seg mat" $ assertAcc (scanl1Seg (+) (use mat) (use seg))
                          ]
                        tScanl'Seg = bgroup "scanl'Seg"
                          [ --testCase "scanl'Seg mat" $ assertAcc (scanl'Seg (+) 0 (use mat) (use seg))
                          ]
                        tPrescanlSeg = bgroup "prescanlSeg"
                          [ --testCase "prescanlSeg mat" $ assertAcc (prescanlSeg (+) 0 (use mat) (use seg))
                          ]
                        tPostscanlSeg = bgroup "postscanlSeg"
                          [ --testCase "postscanlSeg mat" $ assertAcc (postscanlSeg (+) 0 (use mat) (use seg))
                          ]
                        tScanrSeg = bgroup "scanrSeg"
                          [ --testCase "scanrSeg mat" $ assertAcc (scanrSeg (+) 0 (use mat) (use seg))
                          ]
                        tScanr1Seg = bgroup "scanr1Seg"
                          [ --testCase "scanr1Seg mat" $ assertAcc (scanr1Seg (+) (use mat) (use seg))
                          ]
                        tScanr'Seg = bgroup "scanr'Seg"
                          [ --testCase "scanr'Seg mat" $ assertAcc (scanr'Seg (+) 0 (use mat) (use seg))
                          ]
                        tPrescanrSeg = bgroup "prescanrSeg"
                          [ --testCase "prescanrSeg mat" $ assertAcc (prescanrSeg (+) 0 (use mat) (use seg))
                          ]
                        tPostscanrSeg = bgroup "postscanrSeg"
                          [ --testCase "postscanrSeg mat" $ assertAcc (postscanrSeg (+) 0 (use mat) (use seg))
                          ]

        tStencils :: Benchmark
        tStencils = bgroup "Stencils"
          [ tStencil
          ]
          where mat = fromList (Z :. 4 :. 10) [1,2,3,4,5,6,7,8,9,10,1,1,1,1,1,2,2,2,2,2,2,4,6,8,10,12,14,16,18,20,1,3,5,7,9,11,13,15,17,19] :: Matrix Float
                convolve5x1 :: Num a => [Exp a] -> Stencil5x1 a -> Exp a
                convolve5x1 kernel (_, (a,b,c,d,e), _)
                  = Prelude.sum $ Prelude.zipWith (*) kernel [a,b,c,d,e]
                convolve1x5 :: Num a => [Exp a] -> Stencil1x5 a -> Exp a
                convolve1x5 kernel ((_,a,_), (_,b,_), (_,c,_), (_,d,_), (_,e,_))
                  = Prelude.sum $ Prelude.zipWith (*) kernel [a,b,c,d,e]
                gaussian = [0.06136,0.24477,0.38774,0.24477,0.06136] :: [Exp Float]
                blur :: Acc (Matrix Float) -> Acc (Matrix Float)
                blur = stencil (convolve5x1 gaussian) clamp
                     . stencil (convolve1x5 gaussian) clamp
                tStencil = bgroup "stencil"
                  [ testCase "stencil blur" $ assertAcc $ blur $ use mat
                  ]

-- | Test tree corresponding to section "The Accelerate Expression Language" of the Accelerate documentation.
tAccelerateExpressionLanguage :: Benchmark
tAccelerateExpressionLanguage = bgroup "The Accelerate Expression Language"
  [ tBasicTypes,
    tNumTypes,
    tScalarOperations,
    tPlainArrays
  ]
  where tBasicTypes :: Benchmark
        tBasicTypes = bgroup "Basic type classes"
          [ tEq,
            tOrd
          ]
          where vec1 = fromList (Z:.1) [1] :: Vector Int64
                tEq = bgroup "Eq"
                  [ tEquals,
                    tNotEquals
                  ]
                  where tEquals = bgroup "equals (==)"
                          [ testCase "eqInt True" $ assertAcc (map (== 1) (use vec1)),
                            testCase "eqInt False" $ assertAcc (map (== 0) (use vec1))
                          ]
                        tNotEquals = bgroup "not equals (/=)"
                          [ testCase "neqInt True" $ assertAcc (map (/= 1) (use vec1)),
                            testCase "neqInt False" $ assertAcc (map (/= 0) (use vec1))
                          ]
                tOrd = bgroup "Ord"
                  [ tLessThan,
                    tLessThanEquals,
                    tGreaterThan,
                    tGreaterThanEquals,
                    tMin,
                    tMax,
                    tCompare
                  ]
                  where tLessThan = bgroup "less than (<)"
                          [ testCase "ltInt True" $ assertAcc (map (< 2) (use vec1)),
                            testCase "ltInt False" $ assertAcc (map (< 1) (use vec1))
                          ]
                        tLessThanEquals = bgroup "less than equals (<=)"
                          [ testCase "lteInt True" $ assertAcc (map (<= 1) (use vec1)),
                            testCase "lteInt False" $ assertAcc (map (<= 0) (use vec1))
                          ]
                        tGreaterThan = bgroup "greater than (>)"
                          [ testCase "gtInt True" $ assertAcc (map (> 0) (use vec1)),
                            testCase "gtInt False" $ assertAcc (map (> 1) (use vec1))
                          ]
                        tGreaterThanEquals = bgroup "greater than equals (>=)"
                          [ testCase "gteInt True" $ assertAcc (map (>= 1) (use vec1)),
                            testCase "gteInt False" $ assertAcc (map (>= 2) (use vec1))
                          ]
                        tMin = bgroup "min"
                          [ testCase "minInt l" $ assertAcc (map (min 0) (use vec1)),
                            testCase "minInt r" $ assertAcc (map (min 2) (use vec1))
                          ]
                        tMax = bgroup "max"
                          [ testCase "maxInt l" $ assertAcc (map (max 2) (use vec1)),
                            testCase "maxInt r" $ assertAcc (map (max 0) (use vec1))
                          ]
                        tCompare = bgroup "compare"
                          [ testCase "compareInt l" $ assertAcc (map (compare 0) (use vec1)),
                            testCase "compareInt r" $ assertAcc (map (compare 2) (use vec1))
                          ]

        tNumTypes :: Benchmark
        tNumTypes = bgroup "Numeric type classes"
          [ tNum,
            tRational,
            tFractional,
            tFloating,
            tRealFrac,
            tRealFloat
          ]
          where vec1 = fromList (Z:.10) [1..] :: Vector Int32
                vec1' = fromList (Z:.10) [1..] :: Vector Float
                tNum = bgroup "Num"
                  [
                    tPlus,
                    tMinus,
                    tTimes,
                    tNegate,
                    tAbs,
                    tSignum,
                    tQuot,
                    tRem,
                    tDiv,
                    tMod,
                    tQuotRem,
                    tDivMod
                  ]
                  where tPlus = bgroup "plus (+)"
                          [ testCase "plus" $ assertAcc (map (+ 1) (use vec1))
                          ]
                        tMinus = bgroup "minus (-)"
                          [ testCase "minus" $ assertAcc (map (\x -> x - 1) (use vec1) :: Acc (Vector Int32))
                          ]
                        tTimes = bgroup "times (*)"
                          [ testCase "times" $ assertAcc (map (* 2) (use vec1))
                          ]
                        tNegate = bgroup "negate"
                          [ testCase "negate" $ assertAcc (map negate (use vec1))
                          ]
                        tAbs = bgroup "abs"
                          [ testCase "abs" $ assertAcc (map abs (use vec1))
                          ]
                        tSignum = bgroup "signum"
                          [ testCase "signum" $ assertAcc (map signum (use vec1))
                          ]
                        tQuot = bgroup "quot"
                          [ testCase "quot" $ assertAcc (map (quot 2) (use vec1))
                          ]
                        tRem = bgroup "rem"
                          [ testCase "rem" $ assertAcc (map (rem 2) (use vec1))
                          ]
                        tDiv = bgroup "div"
                          [ testCase "div" $ assertAcc (map (div 2) (use vec1))
                          ]
                        tMod = bgroup "mod"
                          [ testCase "mod" $ assertAcc (map (mod 2) (use vec1))
                          ]
                        tQuotRem = bgroup "quotRem"
                          [ testCase "quotRem" $ assertAcc (map (Prelude.fst . quotRem 2) (use vec1)),
                            testCase "quotRem" $ assertAcc (map (Prelude.snd . quotRem 2) (use vec1))
                          ]
                        tDivMod = bgroup "divMod"
                          [ testCase "divMod" $ assertAcc (map (Prelude.fst . divMod 2) (use vec1)),
                            testCase "divMod" $ assertAcc (map (Prelude.snd . divMod 2) (use vec1))
                          ]

                tRational = bgroup "Rational"
                  [ tToRational,
                    tFromRational
                  ]
                  where tToRational = bgroup "toRational"
                          [ testCase "toRational" $ assertAcc (map toRational (use vec1) :: Acc (Vector (Ratio Int64)))
                          ]
                        tFromRational = bgroup "fromRational"
                          [ --testCase "fromRational" $ assertAcc (map fromRational (use vecRatio1)),
                            --testCase "fromRational" $ assertAcc (map fromRational (use vecRatio1))
                          ]

                tFractional = bgroup "Fractional"
                  [ tDivide,
                    tRecip
                  ]
                  where tDivide = bgroup "divide (/)"
                          [ testCase "divide" $ assertAcc (map (/ (2 :: Exp Float)) (use vec1'))
                          ]
                        tRecip = bgroup "recip"
                          [ testCase "recip" $ assertAcc (map recip (use vec1'))
                          ]

                tFloating = bgroup "Floating"
                  [ tPi,
                    tSin,
                    tCos,
                    tTan,
                    tAsin,
                    tAcos,
                    tAtan,
                    tSinh,
                    tCosh,
                    tTanh,
                    tAsinh,
                    tAcosh,
                    tAtanh,
                    tExp,
                    tSqrt,
                    tLog,
                    tFPow,
                    tLogBase
                  ]
                  where vec1 = fromList (Z:.1) [1] :: Vector Int64
                        tPi = bgroup "pi"
                          [ -- testCase "pi" $ assertAcc (map pi (use vec1') :: Acc (Vector Float))
                          ]
                        tSin = bgroup "sin"
                          [ testCase "sin" $ assertAcc (map sin (use vec1'))
                          ]
                        tCos = bgroup "cos"
                          [ testCase "cos" $ assertAcc (map cos (use vec1'))
                          ]
                        tTan = bgroup "tan"
                          [ testCase "tan" $ assertAcc (map tan (use vec1'))
                          ]
                        tAsin = bgroup "asin"
                          [ --testCase "asin" $ assertAcc' (map asin (use vec1'))
                              -- Contains NaN, which when compared is always incorrect test. 
                              -- manually tested: correctly returns [1.5707964,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN]
                          ]
                        tAcos = bgroup "acos"
                          [ --testCase "acos" $ assertAcc (map acos (use vec1'))
                                -- Contains NaN, which when compared is always incorrect test. 
                                -- manually tested: correctly returns [0.0,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN]
                          ]
                        tAtan = bgroup "atan"
                          [ testCase "atan" $ assertAcc (map atan (use vec1'))
                          ]
                        tSinh = bgroup "sinh"
                          [ testCase "sinh" $ assertAcc (map sinh (use vec1'))
                          ]
                        tCosh = bgroup "cosh"
                          [ testCase "cosh" $ assertAcc (map cosh (use vec1'))
                          ]
                        tTanh = bgroup "tanh"
                          [ testCase "tanh" $ assertAcc (map tanh (use vec1'))
                          ]
                        tAsinh = bgroup "asinh"
                          [ testCase "asinh" $ assertAcc (map asinh (use vec1'))
                          ]
                        tAcosh = bgroup "acosh"
                          [ testCase "acosh" $ assertAcc (map acosh (use vec1'))
                          ]
                        tAtanh = bgroup "atanh"
                          [ --testCase "atanh" $ assertAcc (map atanh (use vec1'))
                              -- Contains NaN, which when compared is always incorrect test. 
                              -- manually tested: correctly returns [Infinity,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN]
                          ]
                        tExp = bgroup "exp"
                          [ testCase "exp" $ assertAcc (map exp (use vec1'))
                          ]
                        tSqrt = bgroup "sqrt"
                          [ testCase "sqrt" $ assertAcc (map sqrt (use vec1'))
                          ]
                        tLog = bgroup "log"
                          [ testCase "log" $ assertAcc (map log (use vec1'))
                          ]
                        tFPow = bgroup "(**)"
                          [ testCase "(**)" $ assertAcc (map (** (2 :: Exp Float)) (use vec1'))
                          ]
                        tLogBase = bgroup "logBase"
                          [ testCase "logBase" $ assertAcc (map (logBase (2 :: Exp Float)) (use vec1'))
                          ]


                tRealFrac = bgroup "RealFrac"
                  [ tProperFraction, 
                    -- tTruncate, FAILS
                    tRound,
                    tCeiling,
                    tFloor,
                    tDiv',
                    tMod',
                    tDivMod'
                  ]
                  where tProperFraction = bgroup "properFraction"
                          [ -- testCase "properFraction" $ assertAcc (map (Prelude.snd . properFraction) (use (fromList (Z:.5) [0.2, 0.3, 0.567, 0.8, 3.4])) :: Acc (Vector Float))
                          ]
                        tTruncate = bgroup "truncate"
                          [ testCase "truncate" $ assertAcc (map truncate (use vec1') :: Acc (Vector Int64))
                          ]
                        tRound = bgroup "round"
                          [ testCase "truncate" $ assertAcc (map round (use vec1') :: Acc (Vector Int64))
                          ]
                        tCeiling = bgroup "ceiling"
                          [ testCase "ceiling" $ assertAcc (map ceiling (use vec1') :: Acc (Vector Int64))
                          ]
                        tFloor = bgroup "floor"
                          [ testCase "floor" $ assertAcc (map floor (use vec1') :: Acc (Vector Int64))
                          ]
                        tDiv' = bgroup "div"
                          [ testCase "div" $ assertAcc (map (div (2 :: Exp Int32)) (use vec1) :: Acc (Vector Int32))
                          ]
                        tMod' = bgroup "mod"
                          [ testCase "mod" $ assertAcc (map (mod (2 :: Exp Int32)) (use vec1) :: Acc (Vector Int32))
                          ]
                        tDivMod' = bgroup "divMod"
                          [ --testCase "divMod" $ assertAcc (map (Data.Array.Accelerate.fst . divMod' (2 :: Exp Int64)) (use vec1) :: Acc (Vector (Int64, Int64)))
                          ]

                tRealFloat = bgroup "Real"
                  [ tFloatRadix,
                    tFloatDigits,
                    tFloatRange,
                    -- tDecodeFloat, FAILS
                    tEncodeFloat,
                    -- tExponent, FAILS
                    -- tSignificand, FAILS
                    -- tScaleFloat, FAILS
                    tIsNaN,
                    tIsInfinite,
                    -- tIsDenormalized, FAILS
                    -- tIsNegativeZero, FAILS
                    tIsIEEE,
                    tAtan2
                  ]
                  where tFloatRadix = bgroup "floatRadix"
                          [ testCase "floatRadix" $ assertAcc (map floatRadix (use vec1'))
                          ]
                        tFloatDigits = bgroup "floatDigits"
                          [ testCase "floatDigits" $ assertAcc (map floatDigits (use vec1'))
                          ]
                        tFloatRange = bgroup "floatRange"
                          [ testCase "floatRange" $ assertAcc (map (Prelude.fst . floatRange) (use vec1'))
                          ]
                        tDecodeFloat = bgroup "decodeFloat"
                          [ testCase "decode" $ assertAcc (map (Prelude.fst . decodeFloat) (use vec1'))
                          ]
                        tEncodeFloat = bgroup "encodeFloat"
                          [ testCase "encode" $ assertAcc (map (encodeFloat 1) (use (fromList (Z:.10) [1..10] :: Vector Int)) :: Acc (Vector Float))
                          ]
                        tExponent = bgroup "exponent"
                          [ testCase "exponent" $ assertAcc (map exponent (use vec1'))
                          ]
                        tSignificand = bgroup "significand"
                          [ testCase "significand" $ assertAcc (map significand (use vec1'))
                          ]
                        tScaleFloat = bgroup "scale"
                          [ testCase "scale" $ assertAcc (map (scaleFloat 1) (use vec1'))
                          ]
                        tIsNaN = bgroup "isNaN"
                          [ testCase "isNaN" $ assertAcc (map isNaN (use vec1'))
                          ]
                        tIsInfinite = bgroup "isInfinite"
                          [ testCase "isInfinite" $ assertAcc (map isInfinite (use vec1'))
                          ]
                        tIsDenormalized = bgroup "isDenormalized"
                          [ testCase "isDenormalized" $ assertAcc (map isDenormalized (use vec1'))
                          ]
                        tIsNegativeZero = bgroup "isNegativeZero"
                          [ testCase "isNegativeZero" $ assertAcc (map isNegativeZero (use vec1'))
                          ]
                        tIsIEEE = bgroup "isIEEE"
                          [ testCase "isIEEE" $ assertAcc (map isIEEE (use vec1'))
                          ]
                        tAtan2 = bgroup "atan2"
                          [ testCase "atan2" $ assertAcc (map (atan2 1) (use vec1'))
                          ]

        tScalarOperations :: Benchmark
        tScalarOperations = bgroup "Scalar Operations"
          [ tIntroduction,
            tTuples,
            tFlowControl,
            tScalarReduction,
            tLogicalOperations,
            tNumericOperations,
            tShapeManipulation,
            tConversions
          ]
          where tIntroduction = bgroup "Introduction"
                  [ tConstant
                  ]
                  where vec = fromList (Z:.5) [1..5] :: Vector Int64
                        tConstant = bgroup "constant"
                          [ testCase "constant" $ assertAcc (map (const (constant (1 :: Int))) (use vec))
                          ]
                tTuples = bgroup "Tuples"
                  [ tFst,
                    tAfst,
                    tSnd,
                    tAsnd,
                    tCurry,
                    tUncurry
                  ]
                  where ones = fromList (Z:.5) [1..] :: Vector Int64
                        zeroes = fromList (Z:.5) [0..] :: Vector Int64
                        tFst = bgroup "fst"
                          [ testCase "fst" $ assertAcc (map Data.Array.Accelerate.fst $ zip (use ones) (use zeroes))
                          ]
                        tAfst = bgroup "afst"
                          [ --testCase "afst" $ assertAcc (map afst $ zip (use ones) (use zeroes))
                          ]
                        tSnd = bgroup "snd"
                          [ testCase "snd" $ assertAcc (map Data.Array.Accelerate.snd $ zip (use ones) (use zeroes))
                          ]
                        tAsnd = bgroup "asnd"
                          [ --testCase "asnd" $ assertAcc (map asnd $ zip (use ones) (use zeroes))
                          ]
                        tCurry = bgroup "curry"
                          [ --testCase "curry" $ assertAcc (map (curry (+)) $ zip (use ones) (use zeroes))
                          ]
                        tUncurry = bgroup "uncurry"
                          [ testCase "uncurry" $ assertAcc (map (uncurry (+)) $ zip (use ones) (use zeroes))
                          ]

                tFlowControl = bgroup "Flow Control"
                  [ tQuestionMark,
                    tMatch,
                    tCond,
                    tWhile,
                    tIterate
                  ]
                  where tQuestionMark = bgroup "questionMark"
                          [ --testCase "questionMark" $ assertAcc (Data.Array.Accelerate.zipWith (?) (use (fromList (Z:.3) [True, False, True] :: Vector Bool)) (use (fromList (Z:.3) [(1,2), (2, 3), (3, 4)] :: Vector (Int, Int))))
                          ]
                        tMatch = bgroup "match"
                          [ --
                          ]
                        tCond = bgroup "cond"
                          [ testCase "cond" $ assertAcc (zipWith3 cond (use (fromList (Z:.3) [True, False, True] :: Vector Bool)) (use (fromList (Z:.3) [1, 2, 3] :: Vector Int)) (use (fromList (Z:.3) [4, 5, 6] :: Vector Int)))
                          ]
                        tWhile = bgroup "while"
                          [ --testCase "while" 
                          ]
                        tIterate = bgroup "iterate"
                          [ -- testCase "iterate" 
                            -- testCase "iterate" 
                          ]
                tScalarReduction = bgroup "Scalar Reduction"
                  [
                    tSfoldl
                  ]
                  where vec1 = fromList (Z:.5:.10) [1..] :: Matrix Int
                        tSfoldl = bgroup "sfoldl"
                          [ -- testCase "sfoldl" $ assertAcc (sfoldl (\(I2 x y -> x + y) 0 (use vec1)))
                          ]
                tLogicalOperations = bgroup "Logical Operations"
                  [
                    tAnd,
                    tOr,
                    tNot
                  ]
                  where vecTrue = fromList (Z:.5:.10) (repeat True) :: Matrix Bool
                        vecFalse = fromList (Z:.5:.10) (repeat False) :: Matrix Bool
                        tAnd = bgroup "and (&&)"
                          [ testCase "and True True" $ assertAcc (Data.Array.Accelerate.zipWith (&&) (use vecTrue) (use vecFalse)),
                            testCase "and True False" $ assertAcc (Data.Array.Accelerate.zipWith (&&) (use vecTrue) (use vecFalse)),
                            testCase "and False True" $ assertAcc (Data.Array.Accelerate.zipWith (&&) (use vecFalse) (use vecTrue)),
                            testCase "and False False" $ assertAcc (Data.Array.Accelerate.zipWith (&&) (use vecFalse) (use vecFalse))
                          ]
                        tOr = bgroup "or (||)"
                          [ testCase "or True True" $ assertAcc (Data.Array.Accelerate.zipWith (||) (use vecTrue) (use vecFalse)),
                            testCase "or True False" $ assertAcc (Data.Array.Accelerate.zipWith (||) (use vecTrue) (use vecFalse)),
                            testCase "or False True" $ assertAcc (Data.Array.Accelerate.zipWith (||) (use vecFalse) (use vecTrue)),
                            testCase "or False False" $ assertAcc (Data.Array.Accelerate.zipWith (||) (use vecFalse) (use vecFalse))
                          ]
                        tNot = bgroup "not"
                          [ testCase "not True" $ assertAcc (map not (use vecTrue)),
                            testCase "not False" $ assertAcc (map not (use vecFalse))
                          ]
                tNumericOperations = bgroup "Numeric Operations"
                  [
                    tSubtract,
                    tEven,
                    tOdd
                    -- tGcd, FAILS
                    -- tLcm,
                    -- tHat,
                    -- tHatHat
                  ]
                  where vec = fromList (Z:.5) [1..5] :: Vector Int64
                        vec' = fromList (Z:.5) [1..5] :: Vector Float
                        tSubtract = bgroup "subtract"
                          [ testCase "subtract" $ assertAcc (map (subtract 1) (use vec))
                          ]
                        tEven = bgroup "even"
                          [ testCase "even" $ assertAcc (map even (use vec))
                          ]
                        tOdd = bgroup "odd"
                          [ testCase "odd" $ assertAcc (map odd (use vec))
                          ]
                        tGcd = bgroup "gcd"
                          [ testCase "gcd" $ assertAcc (map (gcd 2) (use vec))
                          ]
                        tLcm = bgroup "lcm"
                          [ testCase "lcm" $ assertAcc (map (lcm 2) (use vec))
                          ]
                        tHat = bgroup "^"
                          [ testCase "^" $ assertAcc (map (^ (2 :: Exp Int64)) (use vec'))
                          ]
                        tHatHat = bgroup "^^"
                          [ testCase "^^" $ assertAcc (map (^^ (2 :: Exp Int64)) (use vec'))
                          ]
                tShapeManipulation = bgroup "Shape Manipulation"
                  [ tIndex0,
                    tIndex1,
                    tUnindex1,
                    tIndex2,
                    tUnindex2,
                    tIndex3,
                    tUnindex3,
                    tIndexHead,
                    tIndexTail,
                    tToIndex,
                    tFromIndex,
                    tIntersect
                  ]
                  where tIndex0 = bgroup "index0"
                          [ testCase "index0" $ assertAcc (map (const index0) (use (fromList (Z:.3) [1, 2, 3] :: Vector Int)))
                          ]
                        tIndex1 = bgroup "index1"
                          [ --testCase "index1" $ 
                          ]
                        tUnindex1 = bgroup "unindex1"
                          [ --testCase "unindex1" $
                          ]
                        tIndex2 = bgroup "index2"
                          [ --testCase "index2" $ 
                          ]
                        tUnindex2 = bgroup "unindex2"
                          [ --testCase "unindex2" $ 
                          ]
                        tIndex3 = bgroup "index3"
                          [ --testCase "index3" $ 
                          ]
                        tUnindex3 = bgroup "unindex3"
                          [ --testCase "unindex3" $
                          ]
                        tIndexHead = bgroup "indexHead"
                          [ --testCase "indexHead" $
                          ]
                        tIndexTail = bgroup "indexTail"
                          [ --testCase "indexTail" $
                          ]
                        tToIndex = bgroup "toIndex"
                          [ --testCase "toIndex" $ 
                          ]
                        tFromIndex = bgroup "fromIndex"
                          [ --testCase "fromIndex" $
                          ]
                        tIntersect = bgroup "intersect"
                          [ --testCase "intersect" $
                          ]
                tConversions = bgroup "Conversions"
                  [ -- tOrd, FAILS
                    -- tChr,
                    tBoolToInt,
                    tBitcast
                  ]
                  where tOrd = bgroup "ord"
                          [ testCase "ord" $ assertAcc (map ord (use (fromList (Z:.3) ['a', 'b', 'c'] :: Vector Char)))
                          ]
                        tChr = bgroup "chr"
                          [ testCase "chr" $ assertAcc (map chr (use (fromList (Z:.3) [97, 98, 99] :: Vector Int)))
                          ]
                        tBoolToInt = bgroup "boolTo"
                          [ testCase "boolTo" $ assertAcc (map boolToInt (use (fromList (Z:.3) [True, False, True] :: Vector Bool)))
                          ]
                        tBitcast = bgroup "bitcast"
                          [ -- testCase "bitcast" $ assertAcc (map bitcast (use (fromList (Z:.3) [1, 2, 3] :: Vector Int)))
                          ]

        tPlainArrays :: Benchmark
        tPlainArrays = bgroup "Plain Arrays"
          [ tOperations,
            tGettingDataIn
          ]
          where tOperations = bgroup "Operations"
                  [ tArrayRank,
                    tArrayShape,
                    tArraySize,
                    tArrayReshape,
                    tIndexArray,
                    tLinearIndexArray
                  ]
                  where tArrayRank = bgroup "arrayRank" -- Seems to be throwing type errors.
                          [ --testCase "arrayRank" $ assertAcc (use (arrayRank (constant (Z:.1:.2) )))
                          ]
                        tArrayShape = bgroup "arrayShape"
                          [ --testCase "arrayShape" $ assertAcc (use (arrayShape (constant (Z:.1:.2))))
                          ]
                        tArraySize = bgroup "arraySize"
                          [ --testCase "arraySize" $ assertAcc (use (arraySize (constant (Z:.1:.2))))
                          ]
                        tArrayReshape = bgroup "arrayReshape"
                          [ --testCase "arrayReshape" $ assertAcc (use (arrayReshape (constant (Z:.1:.2)) (constant (Z:.1:.2))))
                          ]
                        tIndexArray = bgroup "indexArray"
                          [ --testCase "indexArray" $ assertAcc (use (indexArray (constant (Z:.1:.2)) (constant (Z:.1:.2))))
                          ]
                        tLinearIndexArray = bgroup "linearIndexArray"
                          [ --testCase "linearIndexArray" $ assertAcc (use (linearIndexArray (constant (Z:.1:.2)) 1))
                          ]
                tGettingDataIn = bgroup "Getting Data In"
                  [ tFunctions,
                    tLists
                  ]
                  where tFunctions = bgroup "Functions"
                          [ tFromFunction,
                            tFromFunctionM
                          ]
                          where tFromFunction = bgroup "fromFunction"
                                  [ -- testCase "fromFunction" 
                                  ]
                                tFromFunctionM = bgroup "fromFunctionM"
                                  [ --testCase "fromFunctionM" $ assertAcc (use (fromFunctionM Identity (constant (Z:.1:.2)) (\((I2 x y) -> return (constant (x+y))))))
                                  ]
                        tLists = bgroup "Lists"
                          [ tFromList,
                            tToList
                          ]
                          where tFromList = bgroup "fromList"
                                  [ testCase "fromList" $ assertAcc (use (fromList (Z:.1:.2) [1, 2] :: Array DIM2 Int))
                                  ]
                                tToList = bgroup "toList"
                                  [ -- testCase "toList" $ assertAcc (toList (use (fromList (Z:.1:.2) [1, 2] :: Array DIM2 Int)))
                                  ]

