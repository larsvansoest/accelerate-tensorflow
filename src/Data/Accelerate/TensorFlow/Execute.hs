
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

{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE InstanceSigs #-}
{-# OPTIONS_GHC -fno-warn-orphans #-}


module Data.Accelerate.TensorFlow.Execute where
import Data.Array.Accelerate.Backend
    ( DesugarAcc, Execute(..), Backend(..) )
import Data.Array.Accelerate.AST.Schedule.Sequential
    ( expType,
      Var(Var),
      PreArgs(ArgsNil, (:>:)),
      ExpVars,
      GroundsR,
      GroundR(GroundRbuffer, GroundRscalar),
      ArrayInstr(..),
      SArgs,
      SArg(SArgBuffer, SArgScalar),
      SeqSchedule(..),
      SequentialSchedule(..), SeqScheduleFun (..) )
import Data.Accelerate.TensorFlow.Kernel
    ( TensorKernel(..) )
import Data.Array.Accelerate.AST.Execute ( GFunctionR(..) )
import Data.Array.Accelerate.AST.Schedule
    ( IOFun, Scheduled, reprIsBody )
import Data.Array.Accelerate.Representation.Type
    ( TupR(..), TypeR, mapTupR )
import Data.Accelerate.TensorFlow.Type
    ( VectorType, Type64, VectorTypeDict (..), tfVectorTypeDict, toType64, toType64', fromType64', TensorType )
import Data.IORef ( IORef, newIORef, readIORef, writeIORef )
import Data.Array.Accelerate.Type ( ScalarType, Int64 )
import Data.Array.Accelerate.Array.Buffer
    ( Buffer,
      MutableBuffer,
      bufferToList,
      newBuffer,
      unsafeFreezeBuffer,
      writeBuffer )
import Data.Array.Accelerate.Representation.Shape
    ( ShapeR(..) )
import Data.Array.Accelerate.AST.LeftHandSide
    ( LeftHandSide(..) )
import Data.Array.Accelerate.AST.Environment
    ( prj', Env(Push, Empty), update' )
import Control.Concurrent ( MVar, putMVar )
import Data.Array.Accelerate.AST.Idx ( Idx )
import Data.Array.Accelerate.AST.Kernel
    ( OpenKernelFun(..),
      KernelFun,
      KernelArgR(..) )

import qualified Data.Vector.Storable as S

import qualified TensorFlow.Ops                                     as TF
import qualified TensorFlow.Core                                    as TF
import qualified TensorFlow.Session                                 as TF
import qualified TensorFlow.GenOps.Core                             as TF hiding (shape, placeholder)
import Control.Monad.IO.Class ( MonadIO(liftIO) )
import Data.Array.Accelerate.Interpreter
    ( evalExpM, toBool, EvalArrayInstr(EvalArrayInstr) )
import Data.Array.Accelerate.Pretty.Schedule ( PrettySchedule )
import Data.Accelerate.TensorFlow.Operation ( TensorOp )
import Unsafe.Coerce (unsafeCoerce)
import Data.Array.Accelerate.AST.Operation (GroundVars, PrimBool)
import Prelude hiding (exp)
import Data.Data

-- detect copy implementeren voor het versimpelen voor programma's
data TensorFlow where
  TensorFlow :: TensorFlow

instance (DesugarAcc TensorOp, PrettySchedule SequentialSchedule) => Backend TensorFlow where
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
  Tensor :: VectorType (Type64 a) => ScalarType a -> TF.Shape -> IORef (TensorValue (Type64 a)) -> TensorElement (Buffer a)

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
  | VectorTypeDict <- tfVectorTypeDict st
  = do
  sh <- liftIO $ getShape env shR vars
  ref <- liftIO $ newIORef Nil
  return $ TupRsingle $ Tensor st (TF.Shape sh) ref

executeSeqSchedule _ (Use (st :: ScalarType e) n buffer)
  | VectorTypeDict <- tfVectorTypeDict st
  = do
  let cb :: Buffer (Type64 e)
      cb = unsafeCoerce buffer
  let list :: [Type64 e]
      list = bufferToList (toType64 st) n cb
  ref <- liftIO $ newIORef $ Build $ TF.vector list
  return $ TupRsingle $ Tensor st (TF.Shape [fromIntegral n]) ref

executeSeqSchedule env (Unit (Var st idx))
  | (Scalar _ t) <- prj' idx env
  , VectorTypeDict <- tfVectorTypeDict st
  = do ref <- liftIO $ newIORef $ Build $ TF.scalar (toType64' st t)
       return $ TupRsingle $ Tensor st (TF.Shape []) ref

executeSeqSchedule _ (Unit _) = error "impossible"

executeSeqSchedule _ (Acond var ss ss') = undefined

executeSeqSchedule env (Awhile _ cond exp vars) = undefined -- executeAwhile env cond exp vars

executeAwhile :: TensorEnv env -> SeqScheduleFun TensorKernel env (t -> PrimBool) -> SeqScheduleFun TensorKernel env (t -> t) -> GroundVars env t -> TF.Session (TensorElements t)
executeAwhile env cond@(Slam condLhs (Sbody condSched)) exp@(Slam expLhs (Sbody expSched)) vars =
  do let t = mapTupR (\(Var _ idx) -> prj' idx env) vars
     liftIO $ runTensorElements t
     TupRsingle condElement <- executeSeqSchedule (push env (condLhs, t)) condSched
     liftIO $ runTensorElement condElement
     condValue <- liftIO $ returnTensorElement condElement
     if toBool condValue
      then do t' <- executeSeqSchedule (push env (expLhs, t)) expSched
              let env' = updates env vars t'
              executeAwhile env' cond exp vars
      else return t
executeAwhile _ _ _ _ = error "impossible" -- vraag vrijdag: problemen met NIL ref en loops.

updates :: TensorEnv env -> GroundVars env t -> TensorElements t -> TensorEnv env
updates env TupRunit _ = env
updates env (TupRsingle (Var _ idx)) (TupRsingle t) = update' (const t) idx env
updates env (TupRpair vars1 vars2) (TupRpair t1 t2) = updates (updates env vars1 t1) vars2 t2
updates _ _ _ = error "TupR mismatch"

executeKernelFun :: TensorEnv env -> KernelFun TensorKernel args -> SArgs env args -> TF.Session (TensorElements ())
executeKernelFun = executeOpenKernelFun Empty

executeOpenKernelFun :: TensorEnv env' -> TensorEnv env -> OpenKernelFun TensorKernel env' args -> SArgs env args -> TF.Session (TensorElements ())
executeOpenKernelFun env' env (KernelFunLam (KernelArgRscalar _) fun) ((SArgScalar (Var _ idx)) :>: args)     = executeOpenKernelFun (env' `Push` prj' idx env) env fun args
executeOpenKernelFun env' env (KernelFunLam (KernelArgRbuffer _ _) fun) ((SArgBuffer _ (Var _ idx)) :>: args) = executeOpenKernelFun (env' `Push` prj' idx env) env fun args
executeOpenKernelFun env' _ (KernelFunBody kernel) ArgsNil                                                    = executeKernel env' kernel

executeKernel :: TensorEnv env -> TensorKernel env -> TF.Session (TensorElements ())
executeKernel env (TensorConstant st s (Var _ idx))
  | Tensor _ (TF.Shape sh) ref <- prj' idx env
  = do
  let build = Build $ TF.fill (TF.vector sh) (TF.scalar (toType64' st s))
  liftIO $ writeIORef ref build
  return TupRunit
executeKernel env (TensorVar st (Var _ inIdx) outVar)
  | Scalar _ s <- prj' inIdx env
  = executeKernel env (TensorConstant st s outVar)

executeKernel env (TensorId (Var _ inIdx) (Var _ outIdx))                                    = executeUnaryKernel env inIdx outIdx id
executeKernel env (TensorSelect (Var _ inIdx1) (Var _ inIdx2) (Var _ inIdx3) (Var _ outIdx)) = --do liftIO $ putStrLn "select"
                                                                                                  executeTernaryKernel env inIdx1 inIdx2 inIdx3 outIdx $ \x y z -> TF.select (TF.cast x) y z
executeKernel env (TensorWhere (Var _ inIdx) (Var _ outIdx))                                 = --do liftIO $ putStrLn "where"
                                                                                                  executeUnaryKernel env inIdx outIdx (\x -> TF.reshape (TF.where' x) (TF.vector [-1 :: Int64]))
executeKernel env (TensorGather (Var _ inIdx1) (Var _ inIdx2) (Var _ outIdx))                = --do liftIO $ putStrLn "gather"
                                                                                                  executeBinaryKernel env inIdx1 inIdx2 outIdx TF.gather
executeKernel env (TensorBooleanMask (Var _ inIdx1) (Var _ inIdx2) (Var _ outIdx))           = executeBinaryKernel env inIdx1 inIdx2 outIdx undefined -- Somehow boolean mask is missing!

executeKernel env (TensorAdd (Var _ inIdx1) (Var _ inIdx2) (Var _ outIdx))                   = executeBinaryKernel env inIdx1 inIdx2 outIdx TF.add
executeKernel env (TensorMul (Var _ inIdx1) (Var _ inIdx2) (Var _ outIdx))                   = executeBinaryKernel env inIdx1 inIdx2 outIdx TF.mul
executeKernel env (TensorSub (Var _ inIdx1) (Var _ inIdx2) (Var _ outIdx))                   = executeBinaryKernel env inIdx1 inIdx2 outIdx TF.sub
executeKernel env (TensorNeg (Var _ inIdx) (Var _ outIdx))                                   = executeUnaryKernel env inIdx outIdx TF.neg
executeKernel env (TensorAbs (Var _ inIdx) (Var _ outIdx))                                   = executeUnaryKernel env inIdx outIdx TF.abs
executeKernel env (TensorSign (Var _ inIdx) (Var _ outIdx))                                  = executeUnaryKernel env inIdx outIdx TF.sign

executeKernel env (TensorTruncateDiv (Var _ inIdx1) (Var _ inIdx2) (Var _ outIdx))           = executeBinaryKernel env inIdx1 inIdx2 outIdx TF.truncateDiv
executeKernel env (TensorTruncateMod (Var _ inIdx1) (Var _ inIdx2) (Var _ outIdx))           = executeBinaryKernel env inIdx1 inIdx2 outIdx TF.truncateMod
executeKernel env (TensorRealDiv (Var _ inIdx1) (Var _ inIdx2) (Var _ outIdx))               = executeBinaryKernel env inIdx1 inIdx2 outIdx TF.realDiv

executeKernel env (TensorBitwiseAnd (Var _ inIdx1) (Var _ inIdx2) (Var _ outIdx))            = executeBinaryKernel env inIdx1 inIdx2 outIdx TF.bitwiseAnd
executeKernel env (TensorBitwiseOr (Var _ inIdx1) (Var _ inIdx2) (Var _ outIdx))             = executeBinaryKernel env inIdx1 inIdx2 outIdx TF.bitwiseOr
executeKernel env (TensorBitwiseXor (Var _ inIdx1) (Var _ inIdx2) (Var _ outIdx))            = executeBinaryKernel env inIdx1 inIdx2 outIdx TF.bitwiseXor
executeKernel env (TensorInvert (Var _ inIdx) (Var _ outIdx))                                = executeUnaryKernel env inIdx outIdx TF.invert

executeKernel env (TensorReciprocal (Var _ inIdx) (Var _ outIdx))                            = executeUnaryKernel env inIdx outIdx TF.reciprocal
executeKernel env (TensorSin (Var _ inIdx) (Var _ outIdx))                                   = executeUnaryKernel env inIdx outIdx TF.sin
executeKernel env (TensorCos (Var _ inIdx) (Var _ outIdx))                                   = executeUnaryKernel env inIdx outIdx TF.cos
executeKernel env (TensorTan (Var _ inIdx) (Var _ outIdx))                                   = executeUnaryKernel env inIdx outIdx TF.tan
executeKernel env (TensorAsin (Var _ inIdx) (Var _ outIdx))                                  = executeUnaryKernel env inIdx outIdx TF.asin
executeKernel env (TensorAcos (Var _ inIdx) (Var _ outIdx))                                  = executeUnaryKernel env inIdx outIdx TF.acos
executeKernel env (TensorAtan (Var _ inIdx) (Var _ outIdx))                                  = executeUnaryKernel env inIdx outIdx TF.atan
executeKernel env (TensorSinh (Var _ inIdx) (Var _ outIdx))                                  = executeUnaryKernel env inIdx outIdx TF.sinh
executeKernel env (TensorCosh (Var _ inIdx) (Var _ outIdx))                                  = executeUnaryKernel env inIdx outIdx TF.cosh
executeKernel env (TensorTanh (Var _ inIdx) (Var _ outIdx))                                  = executeUnaryKernel env inIdx outIdx TF.tanh
executeKernel env (TensorAsinh (Var _ inIdx) (Var _ outIdx))                                 = executeUnaryKernel env inIdx outIdx TF.asinh
executeKernel env (TensorAcosh (Var _ inIdx) (Var _ outIdx))                                 = executeUnaryKernel env inIdx outIdx TF.acosh
executeKernel env (TensorAtanh (Var _ inIdx) (Var _ outIdx))                                 = executeUnaryKernel env inIdx outIdx TF.atanh
executeKernel env (TensorExp (Var _ inIdx) (Var _ outIdx))                                   = executeUnaryKernel env inIdx outIdx TF.exp
executeKernel env (TensorSqrt (Var _ inIdx) (Var _ outIdx))                                  = executeUnaryKernel env inIdx outIdx TF.sqrt
executeKernel env (TensorLog (Var _ inIdx) (Var _ outIdx))                                   = executeUnaryKernel env inIdx outIdx TF.log
executeKernel env (TensorLog1p (Var _ inIdx) (Var _ outIdx))                                 = executeUnaryKernel env inIdx outIdx TF.log1p

executeKernel env (TensorAtan2 (Var _ inIdx1) (Var _ inIdx2) (Var _ outIdx))                 = executeBinaryKernel env inIdx1 inIdx2 outIdx TF.atan2

executeKernel env (TensorLess (Var _ inIdx1) (Var _ inIdx2) (Var _ outIdx))                  = executeBinaryKernel env inIdx1 inIdx2 outIdx (\x y -> TF.cast (TF.less x y))
executeKernel env (TensorGreater (Var _ inIdx1) (Var _ inIdx2) (Var _ outIdx))               = executeBinaryKernel env inIdx1 inIdx2 outIdx (\x y -> TF.cast (TF.greater x y))
executeKernel env (TensorLessEqual (Var _ inIdx1) (Var _ inIdx2) (Var _ outIdx))             = executeBinaryKernel env inIdx1 inIdx2 outIdx (\x y -> TF.cast (TF.lessEqual x y))
executeKernel env (TensorGreaterEqual (Var _ inIdx1) (Var _ inIdx2) (Var _ outIdx))          = executeBinaryKernel env inIdx1 inIdx2 outIdx (\x y -> TF.cast (TF.greaterEqual x y))
executeKernel env (TensorEqual (Var _ inIdx1) (Var _ inIdx2) (Var _ outIdx))                 = executeBinaryKernel env inIdx1 inIdx2 outIdx (\x y -> TF.cast (TF.equal x y))
executeKernel env (TensorNotEqual (Var _ inIdx1) (Var _ inIdx2) (Var _ outIdx))              = executeBinaryKernel env inIdx1 inIdx2 outIdx (\x y -> TF.cast (TF.notEqual x y))
executeKernel env (TensorMaximum (Var _ inIdx1) (Var _ inIdx2) (Var _ outIdx))               = executeBinaryKernel env inIdx1 inIdx2 outIdx TF.maximum
executeKernel env (TensorMinimum (Var _ inIdx1) (Var _ inIdx2) (Var _ outIdx))               = executeBinaryKernel env inIdx1 inIdx2 outIdx TF.minimum

executeKernel env (TensorLogicalAnd (Var _ inIdx1) (Var _ inIdx2) (Var _ outIdx))            = executeBinaryKernel env inIdx1 inIdx2 outIdx (\x y -> TF.cast (TF.logicalAnd (TF.cast x) (TF.cast y)))
executeKernel env (TensorLogicalOr (Var _ inIdx1) (Var _ inIdx2) (Var _ outIdx))             = executeBinaryKernel env inIdx1 inIdx2 outIdx (\x y -> TF.cast (TF.logicalOr (TF.cast x) (TF.cast y)))
executeKernel env (TensorLogicalNot (Var _ inIdx) (Var _ outIdx))                            = executeUnaryKernel env inIdx outIdx (TF.cast . TF.logicalNot . TF.cast)

executeKernel env (TensorCast (Var _ inIdx) (Var _ outIdx))                                  = executeUnaryKernel env inIdx outIdx TF.cast

executeKernel _ _ = error "impossible"

equalBinaryDimensions :: TensorEnv env -> Idx env (Buffer a) -> Idx env (Buffer a) -> Idx env (Buffer a) -> IO ()
equalBinaryDimensions env inIdx1 inIdx2 outIdx
  | Tensor st1 _ inRef1 <- prj' inIdx1 env
  , Tensor st2 _ inRef2 <- prj' inIdx2 env
  , Tensor _ outSh _ <- prj' outIdx env
  = do
    setShape st1 outSh inRef1
    setShape st2 outSh inRef2

equalBinaryDimensions _ _ _ _ = error "impossible"

executeUnaryKernel :: TensorEnv env -> Idx env (Buffer a) -> Idx env (Buffer b) -> (TF.Tensor TF.Build (Type64 a) -> TF.Tensor TF.Build (Type64 b)) -> TF.Session (TensorElements ())
executeUnaryKernel env inIdx outIdx tfOp
  | Tensor _ sh1 inRef1 <- prj' inIdx env
  , Tensor _ _ outRef <- prj' outIdx env
  = do
    inValue <- liftIO $ readIORef inRef1
    let inTensor = buildTensorValue sh1 inValue

    liftIO $ writeIORef outRef $ Build $ tfOp inTensor
    return TupRunit
executeUnaryKernel _ _ _ _ = error "impossible"

executeBinaryKernel :: TensorEnv env -> Idx env (Buffer a) -> Idx env (Buffer b) -> Idx env (Buffer c) -> (TF.Tensor TF.Build (Type64 a) -> TF.Tensor TF.Build (Type64 b) -> TF.Tensor TF.Build (Type64 c)) -> TF.Session (TensorElements ())
executeBinaryKernel env inIdx1 inIdx2 outIdx tfOp
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
executeBinaryKernel _ _ _ _ _ = error "impossible"

executeTernaryKernel :: TensorEnv env -> Idx env (Buffer a) -> Idx env (Buffer b) -> Idx env (Buffer c) -> Idx env (Buffer d) -> (TF.Tensor TF.Build (Type64 a) -> TF.Tensor TF.Build (Type64 b) -> TF.Tensor TF.Build (Type64 c) -> TF.Tensor TF.Build (Type64 d)) -> TF.Session (TensorElements ())
executeTernaryKernel env inIdx1 inIdx2 inIdx3 outIdx tfOp
  | Tensor _ sh1 inRef1 <- prj' inIdx1 env
  , Tensor _ sh2 inRef2 <- prj' inIdx2 env
  , Tensor _ sh3 inRef3 <- prj' inIdx3 env
  , Tensor _ _ outRef   <- prj' outIdx env
  = do
    inValue1 <- liftIO $ readIORef inRef1
    inValue2 <- liftIO $ readIORef inRef2
    inValue3 <- liftIO $ readIORef inRef3
    let inTensor1 = buildTensorValue sh1 inValue1
    let inTensor2 = buildTensorValue sh2 inValue2
    let inTensor3 = buildTensorValue sh3 inValue3

    liftIO $ writeIORef outRef $ Build $ tfOp inTensor1 inTensor2 inTensor3
    return TupRunit
executeTernaryKernel _ _ _ _ _ _ = error "impossible"

setShape :: (TensorType a, VectorType (Type64 a), S.Storable (Type64 a), TF.TensorDataType S.Vector (Type64 a)) => ScalarType a -> TF.Shape -> IORef (TensorValue (Type64 a)) -> IO ()
setShape _ (TF.Shape sh) ref = do
  value <- readIORef ref
  writeIORef ref $ Build $ TF.reshape (buildTensorValue (TF.Shape sh) value) (TF.vector sh)

getShape :: TensorEnv env -> ShapeR sh -> ExpVars env sh -> IO [Int64]
getShape _ ShapeRz _                                                   = return []
getShape env (ShapeRsnoc shR) (TupRpair vars (TupRsingle (Var _ idx))) = do
  let dim = case prj' idx env of { Scalar _ sh -> fromIntegral sh }
  dims <- getShape env shR vars
  return (dim:dims)
getShape _ _ _ = error "impossible"

push :: TensorEnv env -> (LeftHandSide s t env env', TensorElements t) -> TensorEnv env'
push env (LeftHandSideWildcard _, _)            = env
push env (LeftHandSideSingle _  , TupRsingle a) = env `Push` a
push env (LeftHandSidePair l1 l2, TupRpair a b) = push env (l1, a) `push` (l2, b)
push _ _                                        = error "Tuple mismatch"

evalArrayInstr :: TensorEnv env -> ArrayInstr env (s -> t) -> s -> TF.SessionT IO t
evalArrayInstr env (Index (Var _ idx)) s     = case prj' idx env of
  Scalar _ _ -> error "impossible"
  Tensor st _ ref -> do
    tensorV <- liftIO $ readIORef ref
    case tensorV of
      Build _ -> error "impossible"
      Vector vec -> return $ fromType64' st $ vec S.! s
      Nil -> error "impossible"
evalArrayInstr env (Parameter (Var _ idx)) _ = case prj' idx env of
  Scalar _ e -> return e
  Tensor {} -> error "impossible"

toBuffer :: forall a. (VectorType (Type64 a)) => ScalarType a -> IORef (TensorValue (Type64 a)) -> IO (Buffer a)
toBuffer st ref = do
  Vector (vec :: S.Vector (Type64 a)) <- readIORef ref
  let n = S.length vec
  (mutableBuffer :: MutableBuffer a) <- newBuffer st n
  liftIO $ S.iforM_ vec (\i a -> writeBuffer st mutableBuffer i (fromType64' st a))
  return $ unsafeFreezeBuffer mutableBuffer

-- | convert a tensorvalue to a build tensor
buildTensorValue :: VectorType a => TF.Shape -> TensorValue a -> TF.Tensor TF.Build a
buildTensorValue _ Nil           = error "can not build TNil"
buildTensorValue _ (Build build) = build
buildTensorValue sh (Vector vec) = TF.constant sh $ S.toList vec

-- | convert all tensor values to vectors
runTensorElements :: TensorElements t -> IO ()
runTensorElements TupRunit         = return ()
runTensorElements (TupRsingle v)   = runTensorElement v
runTensorElements (TupRpair v1 v2) = do runTensorElements v1
                                        runTensorElements v2

returnTensorElement :: TensorElement a -> IO a
returnTensorElement (Scalar _ a) = return a
returnTensorElement (Tensor st _ ref) = toBuffer st ref

returnTensorElements :: TensorElements a -> IO a
returnTensorElements TupRunit        = return ()
returnTensorElements (TupRsingle t)  = returnTensorElement t
returnTensorElements (TupRpair t t') = do elems  <- returnTensorElements t
                                          elems' <- returnTensorElements t'
                                          return (elems, elems')

runTensorRef :: VectorType a => IORef (TensorValue a) -> IO ()
runTensorRef ref = do
  value <- readIORef ref
  case value of
    Build build -> do
      vec <- TF.runSession $ TF.run build
      writeIORef ref $ Vector vec
    Vector _ -> return ()
    Nil -> error "can not run NIL ref"

-- | convert a tensorvalue to a vector
runTensorElement :: TensorElement t -> IO ()
runTensorElement (Scalar _ _) = return ()
runTensorElement (Tensor _ _ ref) = runTensorRef ref