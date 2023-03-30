
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs             #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies      #-}
{-# LANGUAGE ScopedTypeVariables #-}



{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# HLINT ignore "Use record patterns" #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE PolyKinds #-}




module Data.Accelerate.TensorFlow.Execute where
import Data.Array.Accelerate.AST.Schedule.Sequential hiding (Nil)
import Data.Accelerate.TensorFlow.Kernel
import Data.Array.Accelerate hiding (map, sum, (++), fromIntegral, size, Vector, Exp)
import Data.Array.Accelerate.AST.LeftHandSide
import Data.Text.Prettyprint.Doc (viaShow, align, group, vcat, vsep, hsep)
import Data.Array.Accelerate.Pretty.Type (prettyScalarType)
import Data.Array.Accelerate.AST.Kernel
import Data.Array.Accelerate.Interpreter hiding (executeKernelFun, executeKernelFun, Right, Left)

import qualified TensorFlow.Ops                                     as TF
import qualified TensorFlow.Core                                    as TF
import qualified TensorFlow.Tensor                                  as TF
import qualified TensorFlow.Types                                   as TF
import qualified TensorFlow.Session                                 as TF
import Data.Vector (Vector)
import qualified TensorFlow.GenOps.Core                             as TF hiding (shape, placeholder)
import Data.IORef
import Data.Array.Accelerate.Array.Buffer
import Data.Array.Accelerate.AST.Environment hiding (push)
import Data.Array.Accelerate.Representation.Type
import Data.Array.Accelerate.Type
import Data.Array.Accelerate.Representation.Ground
import Data.Array.Accelerate.Analysis.Match
import Control.Monad.IO.Class
import Unsafe.Coerce
import Data.Array.Accelerate.Representation.Shape
import Data.Array.Accelerate.Trafo.Operation.Substitution
import qualified Data.Vector.Storable as S
import Control.Concurrent.MVar
import Data.Accelerate.TensorFlow.Type
import Data.Array.Accelerate.Trafo.Var
import Data.Array.Accelerate.Interpreter hiding (executeKernelFun)
import Data.Array.Accelerate.Interpreter (evalExpM)
import qualified Data.Array.Accelerate.Pretty as Pretty
import Data.Array.Accelerate.Pretty.Operation (prettyBuffer)
import Data.Array.Accelerate.Pretty.Exp (Adoc)
import Data.Array.Accelerate.Representation.Elt (showElt)
import Data.Data (Proxy)
import Data.Proxy (Proxy(..))


-- detect copy implementeren voor het versimpelen voor programma's

type TensorEnv = Env TensorElement

data TensorValue a where
  Build :: TF.Tensor TF.Build a -> TensorValue a
  Vector :: S.Vector a -> TensorValue a
  Nil :: TensorValue a

data TensorElement a where
  Scalar :: TypeR a -> a -> TensorElement a
  Tensor :: (TF.TensorType a, S.Storable a, TF.TensorDataType S.Vector a) => ScalarType a -> TF.Shape -> IORef (TensorValue a) -> TensorElement (Buffer a)

-- vraag: waarom werkt dit?
type family TensorElementType a where
  TensorElementType (Buffer a) = a
  TensorElementType a = a

x :: Int -> Int64
x = fromIntegral

y :: Int64 -> Int
y = fromIntegral

type TensorElements = TupR TensorElement

type family TensorElementsIOFun t where
  TensorElementsIOFun (a -> b) = TensorElements a -> TensorElementsIOFun b
  TensorElementsIOFun t        = IO t

executeSequentialSchedule :: TensorEnv env -> SequentialSchedule TensorKernel env t -> TensorElementsIOFun t
executeSequentialSchedule env (SequentialLam lhs sched) = \values -> executeSequentialSchedule (push env (lhs, values)) sched
executeSequentialSchedule env (SequentialBody sched) = \_ -> do
  x <- TF.runSession $ executeSeqSchedule env sched
  runTensorValues x
  str <- prettyVectors x
  putStrLn $ Pretty.renderForTerminal str
  -- met de MVar moet ik nog iets: type class Execute implementeren

executeSeqSchedule :: TensorEnv env -> SeqSchedule TensorKernel env t -> TF.Session (TensorElements t)
executeSeqSchedule env (Exec _ fun args) = executeKernelFun env fun args

executeSeqSchedule env (Return vars) = return $ mapTupR (\(Var _ idx) -> prj' idx env) vars

executeSeqSchedule env (Compute expr) = do
  value <- evalExpM expr $ EvalArrayInstr $ evalArrayInstr env
  return $ TupRsingle (Scalar (expType expr) value)

executeSeqSchedule env (Alet lhs _ sched sched') = do
  rhs <- executeSeqSchedule env sched
  let env' = push env (lhs, rhs)
  executeSeqSchedule env' sched'

executeSeqSchedule env (Alloc shR st vars)
  | TensorTypeDict <- tensorTypeDict st
  = do
  sh <- liftIO $ getShape env shR vars
  ref <- liftIO $ newIORef Nil
  return $ TupRsingle $ Tensor st sh ref

executeSeqSchedule _ (Use st n buffer)
  | TensorTypeDict <- tensorTypeDict st
  = do
  let sh = TF.Shape [fromIntegral n]
  let build = TF.constant sh $ bufferToList st n buffer
  ref <- liftIO $ newIORef $ Build build
  return $ TupRsingle $ Tensor st sh ref

executeSeqSchedule _ (Unit var) = undefined

executeSeqSchedule _ (Acond var ss ss') = undefined

executeSeqSchedule _ (Awhile tr ssf ssf' tr') = undefined

executeKernelFun :: TensorEnv env -> KernelFun TensorKernel args -> SArgs env args -> TF.Session (TensorElements ())
executeKernelFun = executeOpenKernelFun Empty

executeOpenKernelFun :: TensorEnv env' -> TensorEnv env -> OpenKernelFun TensorKernel env' args -> SArgs env args -> TF.Session (TensorElements ())
executeOpenKernelFun env' env (KernelFunLam (KernelArgRscalar _) fun) ((SArgScalar (Var _ idx)) :>: args)     = executeOpenKernelFun (env' `Push` prj' idx env) env fun args
executeOpenKernelFun env' env (KernelFunLam (KernelArgRbuffer _ _) fun) ((SArgBuffer _ (Var _ idx)) :>: args) = executeOpenKernelFun (env' `Push` prj' idx env) env fun args
executeOpenKernelFun env' _ (KernelFunBody kernel) ArgsNil                                                    = executeKernel env' kernel

executeKernel :: TensorEnv env -> TensorKernel env -> TF.Session (TensorElements ())
executeKernel env (TensorConstant s (Var _ idx))
  | Tensor _ (TF.Shape sh) ref <- prj' idx env
  = do
  liftIO $ writeIORef ref $ Build $ TF.fill (TF.vector sh) (TF.scalar s)
  return TupRunit

executeKernel env (TensorAdd TensorDict (Var _ inIdx1) (Var _ inIdx2) (Var _ outIdx))
  | Tensor _ sh1 inRef1 <- prj' inIdx1 env
  , Tensor _ sh2 inRef2 <- prj' inIdx2 env
  , Tensor _ _ outRef   <- prj' outIdx env
  = do
    inValue1 <- liftIO $ readIORef inRef1
    inValue2 <- liftIO $ readIORef inRef2
    let inTensor1 = buildTensorValue sh1 inValue1
    let inTensor2 = buildTensorValue sh2 inValue2
    
    liftIO $ writeIORef outRef $ Build $ TF.add inTensor1 inTensor2
    return TupRunit

executeKernel env (TensorId (Var _ inIdx) (Var _ outIdx)) 
  | Tensor _ sh inRef <- prj' inIdx env
  , Tensor _ _ outRef <- prj' outIdx env
  = do
  inValue <- liftIO $ readIORef inRef
  liftIO $ writeIORef outRef $ Build $ buildTensorValue sh inValue
  return TupRunit

executeKernel _ _ = error "impossible"

getShape :: TensorEnv env -> ShapeR sh -> ExpVars env sh -> IO TF.Shape
getShape env shR vars = do
  sh <- liftIO $ getShape' env shR vars
  return $ toTFShape shR sh
  where
    getShape' :: TensorEnv env -> ShapeR sh -> ExpVars env sh -> IO sh
    getShape' _ ShapeRz _                                                      = return ()
    getShape' env' (ShapeRsnoc shR') (TupRpair vars1 (TupRsingle (Var _ idx))) = do
      sh' <- getShape' env' shR' vars1
      return (sh', case prj' idx env' of { Scalar _ sh -> sh })
    getShape' _ _ _ = error "impossible"

push :: TensorEnv env -> (LeftHandSide s t env env', TensorElements t) -> TensorEnv env'
push env (LeftHandSideWildcard _, _)            = env
push env (LeftHandSideSingle _  , TupRsingle a) = env `Push` a
push env (LeftHandSidePair l1 l2, TupRpair a b) = push env (l1, a) `push` (l2, b)
push _ _                                        = error "Tuple mismatch"

type TensorBuffers = TupR TensorBuffer
data TensorBuffer t where
  TensorBuffer :: ScalarType t -> Int -> Buffer t -> TensorBuffer t

toTFShape :: ShapeR sh -> sh -> TF.Shape
toTFShape shR sh = TF.Shape $ fromIntegral <$> shapeToList shR sh

fromBuffer :: (TF.TensorType t, S.Storable t, TF.TensorDataType S.Vector t) =>
  ShapeR sh ->
  ScalarType t ->
  sh ->
  Buffer t ->
  TF.Session (S.Vector t)
fromBuffer shR t sh buffer = do
  let shape = toTFShape shR sh
  tensorData <- toTensorData t shape (size shR sh) buffer
  TF.runSession $ do
    x <- TF.placeholder shape
    let feeds = [ TF.feed x tensorData ]
    TF.runWithFeeds feeds $ TF.identity x

toTensorData :: (TF.TensorType t, S.Storable t, TF.TensorDataType S.Vector t) =>
  ScalarType t ->
  TF.Shape ->
  Int ->
  Buffer t ->
  TF.Session (TF.TensorData t)
toTensorData t sh n buffer = do
  let vec = S.fromList (bufferToList t n buffer)
  return $ TF.encodeTensorData sh vec

-- prettyBuffers :: TensorBuffers t -> String
-- prettyBuffers TupRunit        = "()"
-- prettyBuffers (TupRsingle (TensorBuffer st n buffer))  = Pretty.renderForTerminal $ prettyBuffer st n buffer
-- prettyBuffers (TupRpair t t') = "(" ++ prettyBuffers t ++ "," ++ prettyBuffers t' ++ ")"

prettyVectors :: TensorElements t -> IO Adoc
prettyVectors TupRunit                     = return $ viaShow "()"
prettyVectors (TupRsingle (Scalar st s))  = return $ viaShow $ fromString $ showElt st s
prettyVectors (TupRsingle (Tensor st _ ref)) = do
  value <- readIORef ref
  case value of
    Build _    -> error "tensorvalue refers to build, please run tensor values first"
    Nil        -> return $ viaShow "TNil"
    Vector vec -> return $ prettyVector st vec
prettyVectors (TupRpair v1 v2) = do
  doc1 <- prettyVectors v1
  doc2 <- prettyVectors v2
  return $ hsep [ "(", doc1, ",", doc2, ")"]

prettyVector :: ScalarType a -> S.Vector a -> Adoc
prettyVector st vec 
 | TensorTypeDict <- tensorTypeDict st
 = align $ group $ "( " <> vcat (mapTail (", " <>) $ map (fromString . showElt (TupRsingle st)) $ S.toList vec) <> " )"

mapTail :: (a -> a) -> [a] -> [a]
mapTail f (x:xs) = x : map f xs
mapTail _ []     = []

-- -- vraag 2: hulp met buffers maken
-- toBuffers :: TensorElements a -> IO (TupR Buffer a)
-- toBuffers TupRunit         = return TupRunit
-- toBuffers (TupRsingle v)   = do buffer <- toBuffer v
--                                 return $ TupRsingle buffer
-- toBuffers (TupRpair v1 v2) = do buffers1 <- toBuffers v1
--                                 buffers2 <- toBuffers v2
--                                 return (TupRpair buffers1 buffers2)

-- toBuffer :: TensorElement a -> IO a
-- toBuffer (Scalar _ _) = error "impossible"
-- toBuffer (Tensor _ ref)
--   = do
--     value <- readIORef ref
--     case value of
--       Build _    -> error "tensorvalue refers to build, please run tensor values first"
--       Nil        -> error "cannot convert TNil to buffer"
--       Vector vec -> do
--         let n = S.length vec
--         mutableBuffer <- newBuffer st n
--         writeVectorToBuffer st vec mutableBuffer
--         return $ unsafeFreezeBuffer mutableBuffer

writeVectorToBuffer :: S.Storable a => ScalarType a -> S.Vector a -> MutableBuffer a -> IO ()
writeVectorToBuffer st vec buffer = S.iforM_ vec (writeBuffer st buffer)

runTensorValues :: TensorElements t -> IO ()
runTensorValues TupRunit         = return ()
runTensorValues (TupRsingle v)   = runTensorValue v
runTensorValues (TupRpair v1 v2) = do runTensorValues v1
                                      runTensorValues v2

runTensorValue :: TensorElement t -> IO ()
runTensorValue (Scalar _ _) = return ()
runTensorValue (Tensor _ _ ref) = do
    value <- readIORef ref
    case value of
      Build build -> do
        vec <- TF.runSession $ TF.run build
        writeIORef ref $ Vector vec
      Vector _ -> return ()
      Nil -> return ()

evalArrayInstr :: TensorEnv env -> ArrayInstr env (s -> t) -> s -> TF.SessionT IO t -- write case matches
evalArrayInstr env (Index (Var _ idx)) s     = case prj' idx env of
  Scalar _ _ -> error "impossible"
  Tensor _ _ ref -> do
    tensorV <- liftIO $ readIORef ref
    case tensorV of
      Build _ -> undefined
      Vector vec -> return $ vec S.! s
      Nil -> error "impossible"
evalArrayInstr env (Parameter (Var _ idx)) _ = case prj' idx env of
  Scalar _ e -> return e
  Tensor {} -> error "impossible"

buildTensorValue :: (S.Storable a, TF.TensorDataType S.Vector a) => TF.Shape -> TensorValue a -> TF.Tensor TF.Build a
buildTensorValue _ Nil           = error "can not build TNil"
buildTensorValue _ (Build build) = build
buildTensorValue sh (Vector vec) = TF.constant sh $ S.toList vec