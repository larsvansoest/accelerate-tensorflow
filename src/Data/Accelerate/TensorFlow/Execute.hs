
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
      GroundsR,
      GroundR(GroundRbuffer, GroundRscalar),
      ArrayInstr(..),
      SArgs,
      SArg(SArgBuffer, SArgScalar),
      SeqSchedule(..),
      SequentialSchedule(..), SeqScheduleFun (..) )
import Data.Accelerate.TensorFlow.Kernel
    ( TensorKernel(..), TensorArg (..) )
import Data.Array.Accelerate.AST.Execute ( GFunctionR(..) )
import Data.Array.Accelerate.AST.Schedule
    ( IOFun, Scheduled, reprIsBody )
import Data.Array.Accelerate.Representation.Type
    ( TupR(..), TypeR, mapTupR )
import Data.Accelerate.TensorFlow.Type
    ( VectorType, Type64, VectorTypeDict (..), tfVectorTypeDict, toType64, toType64', fromType64' )
import Data.IORef ( IORef, newIORef, readIORef, writeIORef )
import Data.Array.Accelerate.Type ( ScalarType, Int64, scalarTypeInt32, Int32 )
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
import Data.Accelerate.TensorFlow.Operation ( TensorOp, ScatterFun (..) )
import Unsafe.Coerce (unsafeCoerce)
import Data.Array.Accelerate.AST.Operation (GroundVars, PrimBool, Vars)
import Prelude hiding (exp)
import Data.Data
import Data.Array.Accelerate.Pretty.Operation (prettyBuffer)
import qualified Data.Array.Accelerate.Pretty as Pretty
import Data.Array.Accelerate.Representation.Elt (showElt)

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
  Build  :: TF.Shape -> TF.Tensor TF.Build a -> TensorValue a
  Vector :: TF.Shape -> S.Vector a -> TensorValue a
  Nil    :: TF.Shape -> TensorValue a

data TensorElement a where
  Scalar :: TypeR a -> a -> TensorElement a
  Tensor :: VectorType (Type64 a) => ScalarType a -> IORef (TensorValue (Type64 a)) -> TensorElement (Buffer a)

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
  ref <- liftIO $ newIORef (Nil (TF.Shape sh))
  return $ TupRsingle $ Tensor st ref

executeSeqSchedule _ (Use (st :: ScalarType e) n buffer)
  | VectorTypeDict <- tfVectorTypeDict st
  = do
  let cb :: Buffer (Type64 e)
      cb = unsafeCoerce buffer
  let list :: [Type64 e]
      list = bufferToList (toType64 st) n cb
  ref <- liftIO $ newIORef $ Build (TF.Shape [fromIntegral n]) (TF.vector list)
  return $ TupRsingle $ Tensor st ref

executeSeqSchedule env (Unit (Var st idx))
  | (Scalar _ t) <- prj' idx env
  , VectorTypeDict <- tfVectorTypeDict st
  = do ref <- liftIO $ newIORef $ Build (TF.Shape []) (TF.scalar (toType64' st t))
       return $ TupRsingle $ Tensor st ref
executeSeqSchedule _ (Unit _) = error "impossible"

executeSeqSchedule env (Acond (Var _ condIdx) exp1 exp2)
  | (Scalar _ cond) <- prj' condIdx env
  = if toBool cond
      then executeSeqSchedule env exp1
      else executeSeqSchedule env exp2

executeSeqSchedule _ (Awhile _ _ _ _) = error "execution of Awhile disabled" -- executeAwhile env cond exp vars

executeAwhile :: TensorEnv env -> SeqScheduleFun TensorKernel env (t -> PrimBool) -> SeqScheduleFun TensorKernel env (t -> t) -> GroundVars env t -> TF.Session (TensorElements t)
executeAwhile env cond body vars = let t = mapTupR (\(Var _ idx) -> prj' idx env) vars in executeAwhile' env cond body t

executeAwhile' :: TensorEnv env -> SeqScheduleFun TensorKernel env (t -> PrimBool) -> SeqScheduleFun TensorKernel env (t -> t) -> TensorElements t -> TF.Session (TensorElements t)
executeAwhile' env cond@(Slam condLhs (Sbody condSched)) body@(Slam expLhs (Sbody expSched)) t = do
  liftIO $ runTensorElements t
  TupRsingle condElement <- executeSeqSchedule (push env (condLhs, t)) condSched
  liftIO $ runTensorElement condElement
  condValue <- liftIO $ returnTensorElement condElement
  if toBool condValue
   then do t' <- executeSeqSchedule (push env (expLhs, t)) expSched
           executeAwhile' env cond body t'
   else return t
executeAwhile' _ _ _ _ = error "impossible"

executeKernelFun :: TensorEnv env -> KernelFun TensorKernel args -> SArgs env args -> TF.Session (TensorElements ())
executeKernelFun = executeOpenKernelFun Empty

executeOpenKernelFun :: TensorEnv env' -> TensorEnv env -> OpenKernelFun TensorKernel env' args -> SArgs env args -> TF.Session (TensorElements ())
executeOpenKernelFun env' env (KernelFunLam (KernelArgRscalar _) fun) ((SArgScalar (Var _ idx)) :>: args)     = executeOpenKernelFun (env' `Push` prj' idx env) env fun args
executeOpenKernelFun env' env (KernelFunLam (KernelArgRbuffer _ _) fun) ((SArgBuffer _ (Var _ idx)) :>: args) = executeOpenKernelFun (env' `Push` prj' idx env) env fun args
executeOpenKernelFun env' _ (KernelFunBody kernel) ArgsNil                                                    = executeKernel env' kernel


-- executeKernel :: TensorEnv env -> TensorKernel env -> TF.Session (TensorElements ())
-- executeKernel env (TensorConstant (TensorArg shR shVars st (Var _ idx)) s)
--   | Tensor _ ref <- prj' idx env
--   = do
--   sh <- liftIO $ getShape env shR shVars
--   liftIO $ writeIORef ref $ Build (TF.Shape sh) $ TF.fill (TF.vector sh) (TF.scalar (toType64' st s))
--   return TupRunit
-- executeKernel _ (TensorConstant _ _)                  = error "impossible"

-- executeKernel env (TensorVar arg (Var _ idx))
--   | Scalar _ s <- prj' idx env                        = executeKernel env (TensorConstant arg s)
-- executeKernel _ (TensorVar _ _)                       = error "impossible"

-- executeKernel env (TensorId aIn aOut)                 = debugExecuteUnaryKernel env aIn aOut id "id"
-- executeKernel env (TensorSelect aIn1 aIn2 aIn3 aOut)  = debugExecuteTernaryKernel env aIn1 aIn2 aIn3 aOut (TF.select . TF.cast) "select"
-- executeKernel env (TensorWhere aIn aOut)              = debugExecuteUnaryKernel env aIn aOut (\x -> TF.reshape (TF.where' x) (TF.vector [-1 :: Int64])) "where"
-- executeKernel env (TensorGather aIn1 aIn2 aOut)       = debugExecuteBinaryKernel env aIn1 aIn2 aOut TF.gather "gather"
-- executeKernel env TensorBooleanMask {}                = undefined -- TODO: find replacement for TF.booleanMask
-- executeKernel env (TensorCast aIn aOut)               = debugExecuteUnaryKernel env aIn aOut TF.cast "cast"

-- executeKernel env (TensorAdd aIn1 aIn2 aOut)          = debugExecuteBinaryKernel env aIn1 aIn2 aOut TF.add "add"
-- executeKernel env (TensorMul aIn1 aIn2 aOut)          = debugExecuteBinaryKernel env aIn1 aIn2 aOut TF.mul "mul"
-- executeKernel env (TensorSub aIn1 aIn2 aOut)          = debugExecuteBinaryKernel env aIn1 aIn2 aOut TF.sub "sub"
-- executeKernel env (TensorNeg aIn aOut)                = debugExecuteUnaryKernel env aIn aOut TF.neg "neg"
-- executeKernel env (TensorAbs aIn aOut)                = debugExecuteUnaryKernel env aIn aOut TF.abs "abs"
-- executeKernel env (TensorSign aIn aOut)               = debugExecuteUnaryKernel env aIn aOut TF.sign "sign"

-- executeKernel env (TensorTruncateDiv aIn1 aIn2 aOut)  = debugExecuteBinaryKernel env aIn1 aIn2 aOut TF.truncateDiv "truncateDiv"
-- executeKernel env (TensorTruncateMod aIn1 aIn2 aOut)  = debugExecuteBinaryKernel env aIn1 aIn2 aOut TF.truncateMod "truncateMod"
-- executeKernel env (TensorRealDiv aIn1 aIn2 aOut)      = debugExecuteBinaryKernel env aIn1 aIn2 aOut TF.realDiv "realDiv" 

-- executeKernel env (TensorBitwiseAnd aIn1 aIn2 aOut)   = debugExecuteBinaryKernel env aIn1 aIn2 aOut TF.bitwiseAnd "bitwiseAnd"
-- executeKernel env (TensorBitwiseOr aIn1 aIn2 aOut)    = debugExecuteBinaryKernel env aIn1 aIn2 aOut TF.bitwiseOr  "bitwiseOr"
-- executeKernel env (TensorBitwiseXor aIn1 aIn2 aOut)   = debugExecuteBinaryKernel env aIn1 aIn2 aOut TF.bitwiseXor "bitwiseXor"
-- executeKernel env (TensorInvert aIn aOut)             = debugExecuteUnaryKernel env aIn aOut TF.invert  "invert"

-- executeKernel env (TensorReciprocal aIn aOut)         = debugExecuteUnaryKernel env aIn aOut TF.reciprocal  "reciprocal"
-- executeKernel env (TensorSin aIn aOut)                = debugExecuteUnaryKernel env aIn aOut TF.sin "sin"
-- executeKernel env (TensorCos aIn aOut)                = debugExecuteUnaryKernel env aIn aOut TF.cos "cos"
-- executeKernel env (TensorTan aIn aOut)                = debugExecuteUnaryKernel env aIn aOut TF.tan "tan"
-- executeKernel env (TensorAsin aIn aOut)               = debugExecuteUnaryKernel env aIn aOut TF.asin  "asin"
-- executeKernel env (TensorAcos aIn aOut)               = debugExecuteUnaryKernel env aIn aOut TF.acos  "acos"
-- executeKernel env (TensorAtan aIn aOut)               = debugExecuteUnaryKernel env aIn aOut TF.atan  "atan"
-- executeKernel env (TensorSinh aIn aOut)               = debugExecuteUnaryKernel env aIn aOut TF.sinh  "sinh"
-- executeKernel env (TensorCosh aIn aOut)               = debugExecuteUnaryKernel env aIn aOut TF.cosh  "cosh"
-- executeKernel env (TensorTanh aIn aOut)               = debugExecuteUnaryKernel env aIn aOut TF.tanh  "tanh"
-- executeKernel env (TensorAsinh aIn aOut)              = debugExecuteUnaryKernel env aIn aOut TF.asinh "asinh"
-- executeKernel env (TensorAcosh aIn aOut)              = debugExecuteUnaryKernel env aIn aOut TF.acosh "acosh"
-- executeKernel env (TensorAtanh aIn aOut)              = debugExecuteUnaryKernel env aIn aOut TF.atanh "atanh"
-- executeKernel env (TensorExp aIn aOut)                = debugExecuteUnaryKernel env aIn aOut TF.exp "exp"
-- executeKernel env (TensorSqrt aIn aOut)               = debugExecuteUnaryKernel env aIn aOut TF.sqrt  "sqrt"
-- executeKernel env (TensorLog aIn aOut)                = debugExecuteUnaryKernel env aIn aOut TF.log "log"
-- executeKernel env (TensorPow aIn1 aIn2 aOut)          = debugExecuteBinaryKernel env aIn1 aIn2 aOut TF.pow  "pow"
-- executeKernel env (TensorLog1p aIn aOut)              = debugExecuteUnaryKernel env aIn aOut TF.log1p "log1p"
-- executeKernel env (TensorAtan2 aIn1 aIn2 aOut)        = debugExecuteBinaryKernel env aIn1 aIn2 aOut TF.atan2  "atan2"

-- executeKernel env (TensorRound aIn aOut)              = debugExecuteUnaryKernel env aIn aOut (TF.cast . TF.round) "round"
-- executeKernel env (TensorFloor aIn aOut)              = debugExecuteUnaryKernel env aIn aOut (TF.cast . TF.floor) "floor"
-- executeKernel env (TensorCeil aIn aOut)               = debugExecuteUnaryKernel env aIn aOut (TF.cast . TF.ceil) "ceil"

-- executeKernel env (TensorIsNan aIn aOut)              = debugExecuteUnaryKernel env aIn aOut (TF.cast . TF.isNan) "isNan"
-- executeKernel env (TensorIsInf aIn aOut)              = debugExecuteUnaryKernel env aIn aOut (TF.cast . TF.isInf) "isInf"

-- executeKernel env (TensorLess aIn1 aIn2 aOut)         = debugExecuteBinaryKernel env aIn1 aIn2 aOut (\x y -> TF.cast (TF.less x y)) "less"
-- executeKernel env (TensorGreater aIn1 aIn2 aOut)      = debugExecuteBinaryKernel env aIn1 aIn2 aOut (\x y -> TF.cast (TF.greater x y)) "greater"
-- executeKernel env (TensorLessEqual aIn1 aIn2 aOut)    = debugExecuteBinaryKernel env aIn1 aIn2 aOut (\x y -> TF.cast (TF.lessEqual x y)) "lessEqual"
-- executeKernel env (TensorGreaterEqual aIn1 aIn2 aOut) = debugExecuteBinaryKernel env aIn1 aIn2 aOut (\x y -> TF.cast (TF.greaterEqual x y)) "greaterEqual"
-- executeKernel env (TensorEqual aIn1 aIn2 aOut)        = debugExecuteBinaryKernel env aIn1 aIn2 aOut (\x y -> TF.cast (TF.equal x y)) "equal"
-- executeKernel env (TensorNotEqual aIn1 aIn2 aOut)     = debugExecuteBinaryKernel env aIn1 aIn2 aOut (\x y -> TF.cast (TF.notEqual x y)) "notEqual"
-- executeKernel env (TensorMaximum aIn1 aIn2 aOut)      = debugExecuteBinaryKernel env aIn1 aIn2 aOut TF.maximum "maximum"
-- executeKernel env (TensorMinimum aIn1 aIn2 aOut)      = debugExecuteBinaryKernel env aIn1 aIn2 aOut TF.minimum "minimum"

-- executeKernel env (TensorLogicalAnd aIn1 aIn2 aOut)   = debugExecuteBinaryKernel env aIn1 aIn2 aOut (\x y -> TF.cast (TF.logicalAnd (TF.cast x) (TF.cast y))) "logicalAnd"
-- executeKernel env (TensorLogicalOr aIn1 aIn2 aOut)    = debugExecuteBinaryKernel env aIn1 aIn2 aOut (\x y -> TF.cast (TF.logicalOr (TF.cast x) (TF.cast y))) "logicalOr"
-- executeKernel env (TensorLogicalNot aIn aOut)         = debugExecuteUnaryKernel env aIn aOut (TF.cast . TF.logicalNot . TF.cast) "logicalNot"

executeKernel :: TensorEnv env -> TensorKernel env -> TF.Session (TensorElements ())
executeKernel env (TensorConstant (TensorArg shR shVars st (Var _ idx)) s)
  | Tensor _ ref <- prj' idx env
  = do
  sh <- liftIO $ getShape env shR shVars
  liftIO $ writeIORef ref $ Build (TF.Shape sh) $ TF.fill (TF.vector sh) (TF.scalar (toType64' st s))
  return TupRunit
executeKernel _ (TensorConstant _ _)                  = error "impossible"

executeKernel env (TensorVar arg (Var _ idx))
  | Scalar _ s <- prj' idx env                        = executeKernel env (TensorConstant arg s)
executeKernel _ (TensorVar _ _)                       = error "impossible"

executeKernel env (TensorId aIn aOut)                 = executeUnaryKernel env aIn aOut id
executeKernel env (TensorSelect aIn1 aIn2 aIn3 aOut)  = executeTernaryKernel env aIn1 aIn2 aIn3 aOut $ \x y z -> TF.select (TF.cast x) y z
executeKernel env (TensorWhere aIn aOut)              = debugExecuteUnaryKernel env aIn aOut (\x -> TF.reshape (TF.where' x) (TF.vector [-1 :: Int64])) "where"
executeKernel env (TensorGather aIn1 aIn2 aOut)       = debugExecuteBinaryKernel env aIn1 aIn2 aOut TF.gather "gather"
executeKernel env (TensorCast aIn aOut)               = executeUnaryKernel env aIn aOut TF.cast

executeKernel env (TensorScatterAdd scatterFun aMut aIn1 aIn2) 
  = debugExecuteTernaryKernel env aMut aIn1 aIn2 aMut (\x y z -> tfScatterFun x (TF.reshape y (TF.concat (TF.scalar 0) [TF.shape y, TF.vector [1 :: Int32]])) z) "tensorScatter"
    where tfScatterFun = case scatterFun of
            ScatterFunAdd -> TF.tensorScatterAdd
            ScatterFunMin -> TF.tensorScatterMin
            ScatterFunMax -> TF.tensorScatterMax
            ScatterFunSub -> TF.tensorScatterSub
            ScatterFunUpdate -> TF.tensorScatterUpdate

executeKernel env (TensorAdd aIn1 aIn2 aOut)          = executeBinaryKernel env aIn1 aIn2 aOut TF.add
executeKernel env (TensorMul aIn1 aIn2 aOut)          = executeBinaryKernel env aIn1 aIn2 aOut TF.mul
executeKernel env (TensorSub aIn1 aIn2 aOut)          = executeBinaryKernel env aIn1 aIn2 aOut TF.sub
executeKernel env (TensorNeg aIn aOut)                = executeUnaryKernel env aIn aOut TF.neg
executeKernel env (TensorAbs aIn aOut)                = executeUnaryKernel env aIn aOut TF.abs
executeKernel env (TensorSign aIn aOut)               = executeUnaryKernel env aIn aOut TF.sign

executeKernel env (TensorTruncateDiv aIn1 aIn2 aOut)  = executeBinaryKernel env aIn1 aIn2 aOut TF.truncateDiv
executeKernel env (TensorTruncateMod aIn1 aIn2 aOut)  = executeBinaryKernel env aIn1 aIn2 aOut TF.truncateMod
executeKernel env (TensorRealDiv aIn1 aIn2 aOut)      = executeBinaryKernel env aIn1 aIn2 aOut TF.realDiv

executeKernel env (TensorBitwiseAnd aIn1 aIn2 aOut)   = executeBinaryKernel env aIn1 aIn2 aOut TF.bitwiseAnd
executeKernel env (TensorBitwiseOr aIn1 aIn2 aOut)    = executeBinaryKernel env aIn1 aIn2 aOut TF.bitwiseOr
executeKernel env (TensorBitwiseXor aIn1 aIn2 aOut)   = executeBinaryKernel env aIn1 aIn2 aOut TF.bitwiseXor
executeKernel env (TensorInvert aIn aOut)             = executeUnaryKernel env aIn aOut TF.invert

executeKernel env (TensorReciprocal aIn aOut)         = executeUnaryKernel env aIn aOut TF.reciprocal
executeKernel env (TensorSin aIn aOut)                = executeUnaryKernel env aIn aOut TF.sin
executeKernel env (TensorCos aIn aOut)                = executeUnaryKernel env aIn aOut TF.cos
executeKernel env (TensorTan aIn aOut)                = executeUnaryKernel env aIn aOut TF.tan
executeKernel env (TensorAsin aIn aOut)               = executeUnaryKernel env aIn aOut TF.asin
executeKernel env (TensorAcos aIn aOut)               = executeUnaryKernel env aIn aOut TF.acos
executeKernel env (TensorAtan aIn aOut)               = executeUnaryKernel env aIn aOut TF.atan
executeKernel env (TensorSinh aIn aOut)               = executeUnaryKernel env aIn aOut TF.sinh
executeKernel env (TensorCosh aIn aOut)               = executeUnaryKernel env aIn aOut TF.cosh
executeKernel env (TensorTanh aIn aOut)               = executeUnaryKernel env aIn aOut TF.tanh
executeKernel env (TensorAsinh aIn aOut)              = executeUnaryKernel env aIn aOut TF.asinh
executeKernel env (TensorAcosh aIn aOut)              = executeUnaryKernel env aIn aOut TF.acosh
executeKernel env (TensorAtanh aIn aOut)              = executeUnaryKernel env aIn aOut TF.atanh
executeKernel env (TensorExp aIn aOut)                = executeUnaryKernel env aIn aOut TF.exp
executeKernel env (TensorSqrt aIn aOut)               = executeUnaryKernel env aIn aOut TF.sqrt
executeKernel env (TensorLog aIn aOut)                = executeUnaryKernel env aIn aOut TF.log
executeKernel env (TensorPow aIn1 aIn2 aOut)          = executeBinaryKernel env aIn1 aIn2 aOut TF.pow
executeKernel env (TensorLog1p aIn aOut)              = executeUnaryKernel env aIn aOut TF.log1p
executeKernel env (TensorAtan2 aIn1 aIn2 aOut)        = executeBinaryKernel env aIn1 aIn2 aOut TF.atan2

executeKernel env (TensorRound aIn aOut)              = executeUnaryKernel env aIn aOut $ TF.cast . TF.round
executeKernel env (TensorFloor aIn aOut)              = executeUnaryKernel env aIn aOut $ TF.cast . TF.floor
executeKernel env (TensorCeil aIn aOut)               = executeUnaryKernel env aIn aOut $ TF.cast . TF.ceil

executeKernel env (TensorIsNan aIn aOut)              = executeUnaryKernel env aIn aOut $ TF.cast . TF.isNan
executeKernel env (TensorIsInf aIn aOut)              = executeUnaryKernel env aIn aOut $ TF.cast . TF.isInf

executeKernel env (TensorLess aIn1 aIn2 aOut)         = executeBinaryKernel env aIn1 aIn2 aOut (\x y -> TF.cast (TF.less x y))
executeKernel env (TensorGreater aIn1 aIn2 aOut)      = executeBinaryKernel env aIn1 aIn2 aOut (\x y -> TF.cast (TF.greater x y))
executeKernel env (TensorLessEqual aIn1 aIn2 aOut)    = executeBinaryKernel env aIn1 aIn2 aOut (\x y -> TF.cast (TF.lessEqual x y))
executeKernel env (TensorGreaterEqual aIn1 aIn2 aOut) = executeBinaryKernel env aIn1 aIn2 aOut (\x y -> TF.cast (TF.greaterEqual x y))
executeKernel env (TensorEqual aIn1 aIn2 aOut)        = executeBinaryKernel env aIn1 aIn2 aOut (\x y -> TF.cast (TF.equal x y))
executeKernel env (TensorNotEqual aIn1 aIn2 aOut)     = executeBinaryKernel env aIn1 aIn2 aOut (\x y -> TF.cast (TF.notEqual x y))
executeKernel env (TensorMaximum aIn1 aIn2 aOut)      = executeBinaryKernel env aIn1 aIn2 aOut TF.maximum
executeKernel env (TensorMinimum aIn1 aIn2 aOut)      = executeBinaryKernel env aIn1 aIn2 aOut TF.minimum

executeKernel env (TensorLogicalAnd aIn1 aIn2 aOut)   = executeBinaryKernel env aIn1 aIn2 aOut (\x y -> TF.cast (TF.logicalAnd (TF.cast x) (TF.cast y)))
executeKernel env (TensorLogicalOr aIn1 aIn2 aOut)    = executeBinaryKernel env aIn1 aIn2 aOut (\x y -> TF.cast (TF.logicalOr (TF.cast x) (TF.cast y)))
executeKernel env (TensorLogicalNot aIn aOut)         = executeUnaryKernel env aIn aOut (TF.cast . TF.logicalNot . TF.cast)

executeUnaryKernel :: TensorEnv env -> TensorArg env sh a -> TensorArg env sh' b -> (TF.Tensor TF.Build (Type64 a) -> TF.Tensor TF.Build (Type64 b)) -> TF.Session (TensorElements ())
executeUnaryKernel env (TensorArg inShR inShVars _ (Var _ inIdx)) (TensorArg outShR outShVars _ (Var _ outIdx)) tfOp
  | Tensor stIn inRef1 <- prj' inIdx env
  , Tensor _ outRef <- prj' outIdx env
  = do
    inSh <- liftIO $ getShape env inShR inShVars
    inTensor <- buildTensor stIn (TF.Shape inSh) inRef1

    outSh <- liftIO $ getShape env outShR outShVars
    liftIO $ writeIORef outRef $ Build (TF.Shape outSh) $ TF.ensureShape (TF.Shape outSh) $ tfOp inTensor
    return TupRunit
executeUnaryKernel _ _ _ _ = error "impossible"

executeBinaryKernel :: TensorEnv env -> TensorArg env sh a -> TensorArg env sh' b -> TensorArg env sh'' c -> (TF.Tensor TF.Build (Type64 a) -> TF.Tensor TF.Build (Type64 b) -> TF.Tensor TF.Build (Type64 c)) -> TF.Session (TensorElements ())
executeBinaryKernel env (TensorArg inShR1 inShVars1 _ (Var _ inIdx1)) (TensorArg inShR2 inShVars2 _ (Var _ inIdx2)) (TensorArg outShR outShVars _ (Var _ outIdx)) tfOp
  | Tensor stIn1 inRef1 <- prj' inIdx1 env
  , Tensor stIn2 inRef2 <- prj' inIdx2 env
  , Tensor _ outRef <- prj' outIdx env
  = do
    inSh1 <- liftIO $ getShape env inShR1 inShVars1
    inTensor1 <- buildTensor stIn1 (TF.Shape inSh1) inRef1
    inSh2 <- liftIO $ getShape env inShR2 inShVars2
    inTensor2 <- buildTensor stIn2 (TF.Shape inSh2) inRef2

    outSh <- liftIO $ getShape env outShR outShVars
    liftIO $ writeIORef outRef $ Build (TF.Shape outSh) $ TF.ensureShape (TF.Shape outSh) $ tfOp inTensor1 inTensor2
    return TupRunit
executeBinaryKernel _ _ _ _ _ = error "impossible"

debugExecuteUnaryKernel :: TensorEnv env -> TensorArg env sh a -> TensorArg env sh' b -> (TF.Tensor TF.Build (Type64 a) -> TF.Tensor TF.Build (Type64 b)) -> String -> TF.Session (TensorElements ())
debugExecuteUnaryKernel env (TensorArg inShR inShVars _ (Var _ inIdx)) (TensorArg outShR outShVars _ (Var _ outIdx)) tfOp s
  | t1@(Tensor stIn inRef1) <- prj' inIdx env
  , t2@(Tensor _ outRef) <- prj' outIdx env
  = do
    liftIO $ putStrLn s

    liftIO $ putStrLn "t1"
    debugTensorElement t1

    inSh <- liftIO $ getShape env inShR inShVars
    inTensor <- buildTensor stIn (TF.Shape inSh) inRef1

    outSh <- liftIO $ getShape env outShR outShVars
    liftIO $ writeIORef outRef $ Build (TF.Shape outSh) $ TF.ensureShape (TF.Shape outSh) $ tfOp inTensor

    liftIO $ putStrLn "t2"
    debugTensorElement t2

    return TupRunit
debugExecuteUnaryKernel _ _ _ _ _ = error "impossible"

debugExecuteBinaryKernel :: TensorEnv env -> TensorArg env sh a -> TensorArg env sh' b -> TensorArg env sh'' c -> (TF.Tensor TF.Build (Type64 a) -> TF.Tensor TF.Build (Type64 b) -> TF.Tensor TF.Build (Type64 c)) -> String -> TF.Session (TensorElements ())
debugExecuteBinaryKernel env (TensorArg inShR1 inShVars1 _ (Var _ inIdx1)) (TensorArg inShR2 inShVars2 _ (Var _ inIdx2)) (TensorArg outShR outShVars _ (Var _ outIdx)) tfOp s
  | t1@(Tensor stIn1 inRef1) <- prj' inIdx1 env
  , t2@(Tensor stIn2 inRef2) <- prj' inIdx2 env
  , t3@(Tensor _ outRef) <- prj' outIdx env
  = do
    liftIO $ putStrLn s

    liftIO $ putStrLn "t1"
    debugTensorElement t1
    liftIO $ putStrLn "t2"
    debugTensorElement t2

    inSh1 <- liftIO $ getShape env inShR1 inShVars1

    inTensor1 <- buildTensor stIn1 (TF.Shape inSh1) inRef1
    inSh2 <- liftIO $ getShape env inShR2 inShVars2
    inTensor2 <- buildTensor stIn2 (TF.Shape inSh2) inRef2

    outSh <- liftIO $ getShape env outShR outShVars
    liftIO $ writeIORef outRef $ Build (TF.Shape outSh) $ TF.ensureShape (TF.Shape outSh) $ tfOp inTensor1 inTensor2

    liftIO $ putStrLn "t3"
    debugTensorElement t3

    return TupRunit
debugExecuteBinaryKernel _ _ _ _ _ _ = error "impossible"

debugExecuteTernaryKernel :: TensorEnv env -> TensorArg env sh a -> TensorArg env sh' b -> TensorArg env sh'' c -> TensorArg env sh''' d -> (TF.Tensor TF.Build (Type64 a) -> TF.Tensor TF.Build (Type64 b) -> TF.Tensor TF.Build (Type64 c) -> TF.Tensor TF.Build (Type64 d)) -> String -> TF.Session (TensorElements ())
debugExecuteTernaryKernel env (TensorArg inShR1 inShVars1 _ (Var _ inIdx1)) (TensorArg inShR2 inShVars2 _ (Var _ inIdx2)) (TensorArg inShR3 inShVars3 _ (Var _ inIdx3)) (TensorArg outShR outShVars _ (Var _ outIdx)) tfOp s
  | t1@(Tensor stIn1 inRef1) <- prj' inIdx1 env
  , t2@(Tensor stIn2 inRef2) <- prj' inIdx2 env
  , t3@(Tensor stIn3 inRef3) <- prj' inIdx3 env
  , t4@(Tensor _ outRef) <- prj' outIdx env
  = do
    liftIO $ putStrLn s

    liftIO $ putStrLn "t1"
    debugTensorElement t1
    liftIO $ putStrLn "t2"
    debugTensorElement t2
    liftIO $ putStrLn "t3"
    debugTensorElement t3

    inSh1 <- liftIO $ getShape env inShR1 inShVars1

    inTensor1 <- buildTensor stIn1 (TF.Shape inSh1) inRef1
    inSh2 <- liftIO $ getShape env inShR2 inShVars2
    inTensor2 <- buildTensor stIn2 (TF.Shape inSh2) inRef2
    inSh3 <- liftIO $ getShape env inShR3 inShVars3
    inTensor3 <- buildTensor stIn3 (TF.Shape inSh3) inRef3

    outSh <- liftIO $ getShape env outShR outShVars
    liftIO $ writeIORef outRef $ Build (TF.Shape outSh) $ TF.ensureShape (TF.Shape outSh) $ tfOp inTensor1 inTensor2 inTensor3

    liftIO $ putStrLn "t4"
    debugTensorElement t4

    return TupRunit
debugExecuteTernaryKernel _ _ _ _ _ _ _ = error "impossible"

debugTensorElement :: TensorElement t -> TF.Session ()
debugTensorElement (Scalar _ _) = liftIO $ putStrLn "scalar"
debugTensorElement t@(Tensor st ref) = do
  liftIO $ runTensorElement t
  Vector (TF.Shape sh) _ <- liftIO $ readIORef ref
  x <- liftIO $ toBuffer st ref
  liftIO $ print sh
  liftIO $ putStrLn $ Pretty.renderForTerminal $ prettyBuffer st (fromIntegral $ product sh) x
  build <- buildTensor st (TF.Shape sh) ref
  liftIO $ writeIORef ref $ Build (TF.Shape sh) build

debugTensorRef :: VectorType a => ScalarType a -> IORef (TensorValue a) -> TF.Session ()
debugTensorRef st ref = do
  liftIO $ runTensorRef ref
  Vector (TF.Shape sh) vec <- liftIO $ readIORef ref
  liftIO $ print sh
  let s = S.foldl (\a b -> a ++ " " ++ showElt (TupRsingle st) b) "" vec
  liftIO $ print s

executeTernaryKernel :: TensorEnv env -> TensorArg env sh a -> TensorArg env sh' b -> TensorArg env sh'' c -> TensorArg env sh''' d -> (TF.Tensor TF.Build (Type64 a) -> TF.Tensor TF.Build (Type64 b) -> TF.Tensor TF.Build (Type64 c) -> TF.Tensor TF.Build (Type64 d)) -> TF.Session (TensorElements ())
executeTernaryKernel env (TensorArg inShR1 inShVars1 _ (Var _ inIdx1)) (TensorArg inShR2 inShVars2 _ (Var _ inIdx2)) (TensorArg inShR3 inShVars3 _ (Var _ inIdx3)) (TensorArg outShR outShVars _ (Var _ outIdx)) tfOp
  | Tensor stIn1 inRef1 <- prj' inIdx1 env
  , Tensor stIn2 inRef2 <- prj' inIdx2 env
  , Tensor stIn3 inRef3 <- prj' inIdx3 env
  , Tensor _ outRef <- prj' outIdx env
  = do
    inSh1 <- liftIO $ getShape env inShR1 inShVars1
    inTensor1 <- buildTensor stIn1 (TF.Shape inSh1) inRef1
    inSh2 <- liftIO $ getShape env inShR2 inShVars2
    inTensor2 <- buildTensor stIn2 (TF.Shape inSh2) inRef2
    inSh3 <- liftIO $ getShape env inShR3 inShVars3
    inTensor3 <- buildTensor stIn3 (TF.Shape inSh3) inRef3

    outSh <- liftIO $ getShape env outShR outShVars
    liftIO $ writeIORef outRef $ Build (TF.Shape outSh) $ TF.ensureShape (TF.Shape outSh) $ tfOp inTensor1 inTensor2 inTensor3
    return TupRunit
executeTernaryKernel _ _ _ _ _ _ = error "impossible"

getShape :: TensorEnv env -> ShapeR sh -> Vars f env sh -> IO [Int64]
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
  Tensor st ref -> do
    tensorV <- liftIO $ readIORef ref
    case tensorV of
      Build sh build -> do vec <- TF.runSession $ TF.run build
                           liftIO $ writeIORef ref $ Vector sh vec
                           return $ fromType64' st $ vec S.! s
      Vector _ vec -> return $ fromType64' st $ vec S.! s
      Nil _ -> error "impossible"
evalArrayInstr env (Parameter (Var _ idx)) _ = case prj' idx env of
  Scalar _ e -> return e
  Tensor {} -> error "impossible"

toBuffer :: forall a. (VectorType (Type64 a)) => ScalarType a -> IORef (TensorValue (Type64 a)) -> IO (Buffer a)
toBuffer st ref = do
  Vector _ (vec :: S.Vector (Type64 a)) <- readIORef ref
  (mutableBuffer :: MutableBuffer a) <- newBuffer st (S.length vec)
  liftIO $ S.iforM_ vec (\i a -> writeBuffer st mutableBuffer i (fromType64' st a))
  return $ unsafeFreezeBuffer mutableBuffer

-- | convert a tensorvalue to a build tensor
buildTensor :: (VectorType (Type64 a)) => ScalarType a -> TF.Shape -> IORef (TensorValue (Type64 a)) -> TF.Session (TF.Tensor TF.Build (Type64 a))
buildTensor _ (TF.Shape sh) ref = do
  value <- liftIO $ readIORef ref
  case value of
    Build (TF.Shape sh') build -> if length sh == length sh'
      then return build
      else do
        --liftIO $ putStrLn $ "reshaping " ++ show sh' ++ " to " ++ show sh
        --debugTensorElement (Tensor st ref)

        let build' = TF.reshape build (TF.vector sh)
        liftIO $ writeIORef ref $ Build (TF.Shape sh) build'

        -- debugTensorElement (Tensor st ref)
        return build'
    Vector _ vec -> do
      let build = TF.constant (TF.Shape sh) $ S.toList vec
      liftIO $ writeIORef ref $ Build (TF.Shape sh) build
      return build
    Nil _ -> error "can not build TNil"

-- | convert all tensor values to vectors
runTensorElements :: TensorElements t -> IO ()
runTensorElements TupRunit         = return ()
runTensorElements (TupRsingle v)   = runTensorElement v
runTensorElements (TupRpair v1 v2) = do runTensorElements v1
                                        runTensorElements v2

returnTensorElement :: TensorElement a -> IO a
returnTensorElement (Scalar _ a) = return a
returnTensorElement (Tensor st ref) = toBuffer st ref

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
    Build sh build -> do
      vec <- TF.runSession $ TF.run build
      writeIORef ref $ Vector sh vec
    Vector _ _ -> return ()
    Nil _ -> error "can not run NIL ref"

-- | convert a tensorvalue to a vector
runTensorElement :: TensorElement t -> IO ()
runTensorElement (Scalar _ _) = return ()
runTensorElement (Tensor _ ref) = runTensorRef ref
