
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
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE InstanceSigs #-}
{-# OPTIONS_GHC -fno-warn-orphans #-}




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
import Data.Array.Accelerate.AST.Idx
import Data.Array.Accelerate.AST.Schedule
import Data.Array.Accelerate.AST.Schedule.Uniform (inputR)
import Data.Array.Accelerate.Pretty.Schedule.Sequential

-- detect copy implementeren voor het versimpelen voor programma's

data TensorFlow where
  TensorFlow :: TensorFlow

instance Backend TensorFlow where
  type Schedule TensorFlow = SequentialSchedule
  type Kernel TensorFlow = TensorKernel

instance Execute SequentialSchedule TensorKernel where
  executeAfunSchedule :: GFunctionR t -> SequentialSchedule TensorKernel () (Scheduled SequentialSchedule t) -> IOFun (Scheduled SequentialSchedule t)
  executeAfunSchedule gFun sched = runTensorElementsIOFun gFun $ executeSequentialSchedule Empty sched

runTensorElementsIOFun :: GFunctionR t
  -> TensorElementsIOFun (Scheduled SequentialSchedule t)
  -> IOFun (Scheduled SequentialSchedule t)
runTensorElementsIOFun (GFunctionRlam gr gFun@(GFunctionRlam _ _)) tef = runTensorElementsIOFun gFun . tef . input gr
runTensorElementsIOFun (GFunctionRlam gr gFun@(GFunctionRbody gr')) tef
  | Refl <- reprIsBody @SequentialSchedule gr' = runTensorElementsIOFun gFun . tef . input gr
runTensorElementsIOFun (GFunctionRbody gr) tef
  | Refl <- reprIsBody @SequentialSchedule gr = \arg -> do x <- tef
                                                           putMVar arg x

input :: GroundsR t' -> t' -> TupR TensorElement t'
input (TupRsingle (GroundRscalar st)) s = TupRsingle $ Scalar (TupRsingle st) s
input (TupRsingle (GroundRbuffer st)) s = undefined -- TupRsingle $ Scalar (TupRsingle st) s -- dit in IO zetten (FullIOFun)
input (TupRpair _ _) _ = error "impossible"
input TupRunit _ = TupRunit

type TensorEnv = Env TensorElement

data TensorValue a where
  Build :: TF.Tensor TF.Build a -> TensorValue a
  Vector :: S.Vector a -> TensorValue a
  Nil :: TensorValue a

data TensorElement a where
  Scalar :: TypeR a -> a -> TensorElement a
  Tensor :: TensorType a => ScalarType a -> TF.Shape -> IORef (TensorValue a) -> TensorElement (Buffer a)

type TensorElements = TupR TensorElement

type family TensorElementsIOFun t where
  TensorElementsIOFun (MVar a -> ()) = IO a
  TensorElementsIOFun (a -> b)       = TensorElements a -> TensorElementsIOFun b
  TensorElementsIOFun t              = IO t

executeSequentialSchedule :: forall env t. TensorEnv env -> SequentialSchedule TensorKernel env t -> TensorElementsIOFun t
executeSequentialSchedule env (SequentialLam lhs sched@(SequentialLam _ _)) = \values -> executeSequentialSchedule (push env (lhs, values)) sched
executeSequentialSchedule env (SequentialLam lhs sched@(SequentialBody _)) = \values -> executeSequentialSchedule (push env (lhs, values)) sched
executeSequentialSchedule env (SequentialBody sched) = do
  elems <- TF.runSession $ executeSeqSchedule env sched
  runTensorElements elems
  returnTensorElements elems

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
executeKernel env (TensorId (Var _ inIdx) (Var _ outIdx))                          = executeUnaryKernel1 env inIdx outIdx id

executeKernel env (TensorAdd (Var _ inIdx1) (Var _ inIdx2) (Var _ outIdx))         = executeBinaryKernel1 env inIdx1 inIdx2 outIdx TF.add
executeKernel env (TensorMul (Var _ inIdx1) (Var _ inIdx2) (Var _ outIdx))         = executeBinaryKernel1 env inIdx1 inIdx2 outIdx TF.mul
executeKernel env (TensorSub (Var _ inIdx1) (Var _ inIdx2) (Var _ outIdx))         = executeBinaryKernel1 env inIdx1 inIdx2 outIdx TF.sub
executeKernel env (TensorNeg (Var _ inIdx) (Var _ outIdx))                         = executeUnaryKernel1 env inIdx outIdx TF.neg
executeKernel env (TensorAbs (Var _ inIdx) (Var _ outIdx))                         = executeUnaryKernel1 env inIdx outIdx TF.abs
executeKernel env (TensorSign (Var _ inIdx) (Var _ outIdx))                        = executeUnaryKernel1 env inIdx outIdx TF.sign

executeKernel env (TensorTruncateDiv (Var _ inIdx1) (Var _ inIdx2) (Var _ outIdx)) = executeBinaryKernel1 env inIdx1 inIdx2 outIdx TF.truncateDiv
executeKernel env (TensorTruncateMod (Var _ inIdx1) (Var _ inIdx2) (Var _ outIdx)) = executeBinaryKernel1 env inIdx1 inIdx2 outIdx TF.truncateMod

executeKernel _ _ = error "impossible"

executeUnaryKernel1 :: TensorEnv env -> Idx env (Buffer a) -> Idx env (Buffer b) -> (TF.Tensor TF.Build a -> TF.Tensor TF.Build b) -> TF.Session (TensorElements ())
executeUnaryKernel1 env inIdx outIdx tfOp
  | Tensor _ sh1 inRef1 <- prj' inIdx env
  , Tensor _ _ outRef   <- prj' outIdx env
  = do
    inValue <- liftIO $ readIORef inRef1
    let inTensor = buildTensorValue sh1 inValue

    liftIO $ writeIORef outRef $ Build $ tfOp inTensor
    return TupRunit
executeUnaryKernel1 _ _ _ _ = error "impossible"

executeBinaryKernel1 :: TensorEnv env -> Idx env (Buffer a) -> Idx env (Buffer a) -> Idx env (Buffer b) -> (TF.Tensor TF.Build a -> TF.Tensor TF.Build a -> TF.Tensor TF.Build b) -> TF.Session (TensorElements ())
executeBinaryKernel1 env inIdx1 inIdx2 outIdx tfOp
  | Tensor _ sh1 inRef1 <- prj' inIdx1 env
  , Tensor _ sh2 inRef2 <- prj' inIdx2 env
  , Tensor _ _ outRef   <- prj' outIdx env
  = do
    inValue1 <- liftIO $ readIORef inRef1
    inValue2 <- liftIO $ readIORef inRef2
    let inTensor1 = buildTensorValue sh1 inValue1
    let inTensor2 = buildTensorValue sh2 inValue2

    liftIO $ writeIORef outRef $ Build $ tfOp inTensor1 inTensor2
    return TupRunit
executeBinaryKernel1 _ _ _ _ _ = error "impossible"

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

toTFShape :: ShapeR sh -> sh -> TF.Shape
toTFShape shR sh = TF.Shape $ fromIntegral <$> shapeToList shR sh

fromBuffer :: TensorType a =>
  ShapeR sh ->
  ScalarType a ->
  sh ->
  Buffer a ->
  TF.Session (S.Vector a)
fromBuffer shR t sh buffer = do
  let shape = toTFShape shR sh
  tensorData <- toTensorData t shape (size shR sh) buffer
  TF.runSession $ do
    x <- TF.placeholder shape
    let feeds = [ TF.feed x tensorData ]
    TF.runWithFeeds feeds $ TF.identity x

toTensorData :: TensorType a =>
  ScalarType a ->
  TF.Shape ->
  Int ->
  Buffer a ->
  TF.Session (TF.TensorData a)
toTensorData t sh n buffer = do
  let vec = S.fromList (bufferToList t n buffer)
  return $ TF.encodeTensorData sh vec

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

toBuffer :: TensorType a => ScalarType a -> IORef (TensorValue a) -> IO (Buffer a)
toBuffer st ref = do
  runTensorRef ref
  Vector vec <- readIORef ref
  let n = S.length vec
  mutableBuffer <- newBuffer st n
  writeVectorToBuffer st vec mutableBuffer
  return $ unsafeFreezeBuffer mutableBuffer

writeVectorToBuffer :: TensorType a => ScalarType a -> S.Vector a -> MutableBuffer a -> IO ()
writeVectorToBuffer st vec buffer = S.iforM_ vec (writeBuffer st buffer)

-- | convert a tensorvalue to a build tensor
buildTensorValue :: TensorType a => TF.Shape -> TensorValue a -> TF.Tensor TF.Build a
buildTensorValue _ Nil           = error "can not build TNil"
buildTensorValue _ (Build build) = build
buildTensorValue sh (Vector vec) = TF.constant sh $ S.toList vec

-- | convert all tensor values to vectors
runTensorElements :: TensorElements t -> IO ()
runTensorElements TupRunit         = return ()
runTensorElements (TupRsingle v)   = runTensorElement v
runTensorElements (TupRpair v1 v2) = do runTensorElements v1
                                        runTensorElements v2

returnTensorElements :: TensorElements a -> IO a
returnTensorElements TupRunit = return ()
returnTensorElements (TupRsingle t) = case t of
  Scalar _ a -> return a
  Tensor st _ ir -> toBuffer st ir
returnTensorElements (TupRpair t t') = do elems  <- returnTensorElements t
                                          elems' <- returnTensorElements t'
                                          return (elems, elems')

runTensorRef :: TensorType a => IORef (TensorValue a) -> IO ()
runTensorRef ref = do
  value <- readIORef ref
  case value of
    Build build -> do
      vec <- TF.runSession $ TF.run build
      writeIORef ref $ Vector vec
    Vector _ -> return ()
    Nil -> return ()

-- | convert a tensorvalue to a vector
runTensorElement :: TensorElement t -> IO ()
runTensorElement (Scalar _ _) = return ()
runTensorElement (Tensor _ _ ref) = runTensorRef ref